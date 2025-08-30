import faiss
import torch
import os
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
    Tạo một từ điển ánh xạ từ chỉ mục của vector đặc trưng
    đến đường dẫn file .npy và ID video.
    """
    mapping = {}
    idx = 0
    # Đảm bảo các thư mục con được sắp xếp để có thứ tự nhất quán
    for video_id in sorted(os.listdir(embeddings_folder)):
        video_path = os.path.join(embeddings_folder, video_id)
        if os.path.isdir(video_path):
            # Sắp xếp các file .npy trong mỗi thư mục
            for frame_file in sorted(os.listdir(video_path)):
                if frame_file.endswith(".npy"):
                    frame_name_without_ext = os.path.splitext(frame_file)[0]
                    mapping[idx] = {
                        "video_id": video_id,
                        "frame_npy_path": os.path.join(video_path, frame_file),
                        "frame_name": frame_name_without_ext
                    }
                    idx += 1
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
    Tải các vector đặc trưng và xây dựng chỉ mục FAISS.
    """
    embeddings_list = []
    # Lặp qua các file .npy trong thư mục
    for filename in sorted(os.listdir(embeddings_folder)):
        if filename.endswith(".npy"):
            file_path = os.path.join(embeddings_folder, filename)
            embedding = np.load(file_path).astype('float32') # FAISS yêu cầu float32
            embeddings_list.append(embedding)

    if not embeddings_list:
        print("Không tìm thấy file .npy nào.")
        return None, None

    # Gộp tất cả các vector lại thành một mảng NumPy
    embeddings_matrix = np.vstack(embeddings_list)

    d = embeddings_matrix.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings_matrix)
    
    print(f"Đã xây dựng chỉ mục FAISS với {index.ntotal} vector.")
    return index, embeddings_matrix

def search_top_k(query_embedding: np.ndarray, index: faiss.Index, k: int = 5):
    """
    Tìm kiếm k vector gần nhất với vector truy vấn.
    """
    D, I = index.search(query_embedding.reshape(1, -1).astype('float32'), k)
    return D, I
