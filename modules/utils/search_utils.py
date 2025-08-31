import faiss
import torch
import os
import re
import clip
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import pandas as pd

# Các hàm đọc dữ liệu đã thêm vào
def load_video_metadata(metadata_path):
    """Đọc file metadata của video."""
    if not os.path.exists(metadata_path):
        return None
    with open(metadata_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_keyframes_map(map_path):
    """Đọc file ánh xạ keyframes."""
    if not os.path.exists(map_path):
        return None
    return pd.read_csv(map_path)

def load_object_detections(object_path):
    """Đọc file phát hiện đối tượng."""
    if not os.path.exists(object_path):
        return None
    with open(object_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_index_to_path_mapping(embeddings_folder):
    """
    Tạo mapping index -> thông tin (file .npy, row trong file).
    Đảm bảo cùng thứ tự với build_faiss_index.
    """
    mapping = {}
    idx = 0
    for file in sorted(os.listdir(embeddings_folder)):
        if file.endswith(".npy"):
            file_path = os.path.join(embeddings_folder, file)
            arr = np.load(file_path)

            # Nếu chỉ có 1 vector (1D)
            if arr.ndim == 1:
                mapping[idx] = {
                    "frame_npy_path": file_path,
                    "file_row": 0,
                    "frame_name": os.path.splitext(file)[0]
                }
                idx += 1

            # Nếu nhiều vector (2D)
            elif arr.ndim == 2:
                for r in range(arr.shape[0]):
                    mapping[idx] = {
                        "frame_npy_path": file_path,
                        "file_row": r,
                        "frame_name": f"{os.path.splitext(file)[0]}_row{r}"
                    }
                    idx += 1

            else:
                print(f"⚠️ Cảnh báo: {file_path} có shape {arr.shape}, bỏ qua.")
    return mapping

def process_query_text(query_text: str, model, device):
    """
    Tạo vector đặc trưng từ câu truy vấn văn bản bằng mô hình CLIP.
    """
    text_tokens = clip.tokenize([query_text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features.cpu().numpy().astype('float32')

def build_faiss_index(embeddings_folder: str):
    """
    Tải các vector đặc trưng từ tất cả thư mục con và xây dựng chỉ mục FAISS.
    Trả về:
        - index: FAISS index
        - embeddings_matrix: Ma trận vector
        - id_mapping: Danh sách mapping từ index -> file gốc
    """
    embeddings_list = []
    id_mapping = []

    # Duyệt qua toàn bộ thư mục con
    for root, _, files in os.walk(embeddings_folder):
        for filename in sorted(files):
            if filename.endswith(".npy"):
                file_path = os.path.join(root, filename)
                embedding = np.load(file_path).astype('float32')

                # Nếu vector 1D thì reshape thành (1, d)
                if embedding.ndim == 1:
                    embedding = embedding.reshape(1, -1)

                embeddings_list.append(embedding)
                id_mapping.extend([file_path] * embedding.shape[0])

    if not embeddings_list:
        print("Không tìm thấy file .npy nào.")
        return None, None, None

    # Gộp tất cả embeddings thành ma trận
    embeddings_matrix = np.vstack(embeddings_list)

    # Tạo FAISS index
    d = embeddings_matrix.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings_matrix)

    print(f"Đã xây dựng chỉ mục FAISS với {index.ntotal} vector.")
    return index, embeddings_matrix, id_mapping

def search_top_k(query_embedding: np.ndarray, index: faiss.Index, k: int = 5):
    """
    Tìm kiếm k vector gần nhất với vector truy vấn.
    """
    D, I = index.search(query_embedding.reshape(1, -1).astype('float32'), k)
    return D, I

def preprocess_query_text(description: str) -> list[str]:
    """
    Tách query dài thành các câu ngắn hơn (<=77 tokens).
    Ở đây dùng rule-based: tách theo dấu chấm, phẩy.
    """
    candidates = [q.strip() for q in re.split(r"[.,;]", description) if q.strip()]
    if not candidates:
        candidates = [description]
    return candidates


def encode_queries(description, model, device):
    """
    Encode query:
    - Nếu description là string dài => tự động tách thành sub-queries.
    - Nếu description là list => encode từng câu rồi lấy mean.
    """
    if isinstance(description, str):
        sub_queries = preprocess_query_text(description)
    elif isinstance(description, list):
        sub_queries = description
    else:
        raise ValueError("description phải là string hoặc list các string.")

    embeddings = [process_query_text(sub_q, model, device) for sub_q in sub_queries]
    return np.mean(embeddings, axis=0)