import faiss
import torch
import os
import clip
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import pandas as pd

def load_video_metadata(metadata_path):
    """Đọc file metadata của video (L21_V001.json)."""
    with open(metadata_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_keyframes_map(map_path):
    """Đọc file ánh xạ keyframes (L21_V001.csv)."""
    df = pd.read_csv(map_path)
    return df

def load_object_detections(object_path):
    """Đọc file phát hiện đối tượng (001.json)."""
    with open(object_path, 'r', encoding='utf-8') as f:
        return json.load(f)


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
