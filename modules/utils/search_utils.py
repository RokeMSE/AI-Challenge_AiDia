import faiss
import torch
import os
import clip
import numpy as np
import json

def create_index_to_path_mapping(embeddings_folder):
    """
    Recursively creates a map from a flat FAISS index to the original file path.
    """
    mapping = {}
    idx = 0
    # Walk through all subdirectories
    for root, _, files in os.walk(embeddings_folder):
        # Sort files to ensure consistent order
        for frame_file in sorted(files):
            if frame_file.endswith(".npy"):
                video_id = os.path.basename(root)
                frame_name_without_ext = os.path.splitext(frame_file)[0]
                mapping[idx] = {
                    "video_id": video_id,
                    "frame_npy_path": os.path.join(root, frame_file),
                    "frame_name": frame_name_without_ext
                }
                idx += 1
    return mapping

def build_faiss_index(embeddings_folder: str):
    """
    Recursively loads all embeddings and builds a FAISS index.
    """
    embeddings_list = []
    print("Scanning for all .npy embedding files...")
    # Walk through all subdirectories
    for root, _, files in os.walk(embeddings_folder):
        # Sort files to ensure consistent order
        for filename in sorted(files):
            if filename.endswith(".npy"):
                file_path = os.path.join(root, filename)
                embedding = np.load(file_path).astype('float32')
                embeddings_list.append(embedding)

    if not embeddings_list:
        print("No .npy files found in the embeddings folder.")
        return None

    embeddings_matrix = np.vstack(embeddings_list)
    d = embeddings_matrix.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings_matrix)
    
    print(f"FAISS index built successfully with {index.ntotal} total vectors.")
    return index

def search_top_k(query_embedding: np.ndarray, index: faiss.Index, k: int = 5):
    """
    Searches the FAISS index for the top k nearest neighbors.
    """
    D, I = index.search(query_embedding.reshape(1, -1).astype('float32'), k)
    return D, I

def find_best_frame_in_video(text_query, video_id, embeddings_folder, model):
    """
    Finds the single best frame in a specific video for a given text query.
    """
    video_embeddings_path = os.path.join(embeddings_folder, video_id)
    if not os.path.isdir(video_embeddings_path):
        return None, -1

    frame_files = sorted([f for f in os.listdir(video_embeddings_path) if f.endswith('.npy')])
    if not frame_files:
        return None, -1
        
    video_frame_embeddings = np.array([np.load(os.path.join(video_embeddings_path, f)) for f in frame_files])
    
    query_embedding = model.get_text_features([text_query])
    
    # Cosine similarity is the dot product of normalized vectors
    similarities = np.dot(video_frame_embeddings, query_embedding.T).flatten()
    
    best_frame_idx = np.argmax(similarities)
    best_frame_name = os.path.splitext(frame_files[best_frame_idx])[0]
    
    return best_frame_name, similarities[best_frame_idx]