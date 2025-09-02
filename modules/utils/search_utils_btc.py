import faiss
import torch
import os
import clip
import numpy as np
import json

def create_index_to_path_mapping(embeddings_folder):
    """
    Recursively creates a map from a flat FAISS index to the original file path and vector index.
    For BTC data where each .npy file contains a 2D array of vectors.
    """
    mapping = {}
    idx = 0
    
    print("Creating index mapping for BTC embeddings...")
    # Walk through all subdirectories
    for root, _, files in os.walk(embeddings_folder):
        video_id = os.path.basename(root)
        # Sort files to ensure consistent order
        for frame_file in sorted(files):
            if frame_file.endswith(".npy"):
                frame_name_without_ext = os.path.splitext(frame_file)[0]
                npy_path = os.path.join(root, frame_file)
                
                try:
                    # Load the 2D array to get the number of vectors
                    vectors_2d = np.load(npy_path)
                    if vectors_2d.ndim != 2:
                        print(f"Warning: {frame_file} is not a 2D array. Skipping.")
                        continue
                    
                    # Create mapping for each vector in the 2D array
                    num_vectors = vectors_2d.shape[0]
                    for vector_idx in range(num_vectors):
                        mapping[idx] = {
                            "video_id": video_id,
                            "frame_npy_path": npy_path,
                            "frame_name": frame_name_without_ext,
                            "vector_index": vector_idx
                        }
                        idx += 1
                        
                except Exception as e:
                    print(f"Error processing {frame_file}: {e}")
                    continue
    
    print(f"Created mapping for {len(mapping)} total vectors")
    return mapping

def build_faiss_index(embeddings_folder: str):
    """
    Recursively loads all 2D embedding arrays and builds a FAISS index.
    Each row in each 2D array becomes a separate entry in the index.
    """
    embeddings_list = []
    print("Building FAISS index from BTC embeddings...")
    
    # Walk through all subdirectories
    for root, _, files in os.walk(embeddings_folder):
        # Sort files to ensure consistent order
        for filename in sorted(files):
            if filename.endswith(".npy"):
                file_path = os.path.join(root, filename)
                try:
                    # Load the 2D array
                    vectors_2d = np.load(file_path).astype('float32')
                    
                    if vectors_2d.ndim != 2:
                        print(f"Warning: {filename} is not a 2D array (shape: {vectors_2d.shape}). Skipping.")
                        continue
                    
                    # Add each vector (row) to the list
                    for vector in vectors_2d:
                        embeddings_list.append(vector)
                        
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
                    continue

    if not embeddings_list:
        print("No valid 2D .npy files found in the embeddings folder.")
        return None

    # Stack all vectors into a single matrix
    embeddings_matrix = np.vstack(embeddings_list)
    d = embeddings_matrix.shape[1]
    
    # Create FAISS index
    index = faiss.IndexFlatL2(d)
    index.add(embeddings_matrix)
    
    print(f"FAISS index built successfully with {index.ntotal} total vectors from 2D arrays.")
    return index

def search_top_k(query_embedding: np.ndarray, index: faiss.Index, k: int = 5):
    """
    Searches the FAISS index for the top k nearest neighbors.
    """
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    
    D, I = index.search(query_embedding.astype('float32'), k)
    return D, I

def find_best_vectors_in_video(text_query, video_id, embeddings_folder, model, top_k=5):
    """
    Finds the best vectors in a specific video for a given text query.
    For BTC data where each frame file contains multiple vectors.
    """
    video_embeddings_path = os.path.join(embeddings_folder, video_id)
    if not os.path.isdir(video_embeddings_path):
        return []

    frame_files = sorted([f for f in os.listdir(video_embeddings_path) if f.endswith('.npy')])
    if not frame_files:
        return []
    
    # Collect all vectors with their metadata
    all_vectors = []
    vector_metadata = []
    
    for frame_file in frame_files:
        frame_name = os.path.splitext(frame_file)[0]
        file_path = os.path.join(video_embeddings_path, frame_file)
        
        try:
            vectors_2d = np.load(file_path).astype('float32')
            
            if vectors_2d.ndim != 2:
                print(f"Warning: {frame_file} is not a 2D array. Skipping.")
                continue
            
            # Add each vector with its metadata
            for vector_idx, vector in enumerate(vectors_2d):
                all_vectors.append(vector)
                vector_metadata.append({
                    'frame_name': frame_name,
                    'vector_index': vector_idx
                })
                
        except Exception as e:
            print(f"Error loading {frame_file}: {e}")
            continue
    
    if not all_vectors:
        return []
    
    # Stack all vectors and compute similarities
    video_embeddings_matrix = np.vstack(all_vectors)
    
    # Get query embedding (assuming model has encode method like SentenceTransformer)
    if hasattr(model, 'encode'):
        query_embedding = model.encode([text_query])[0]
    else:
        # Fallback for CLIP-like models
        query_embedding = model.get_text_features([text_query])[0]
    
    # Compute cosine similarities
    # Normalize vectors for cosine similarity
    video_embeddings_norm = video_embeddings_matrix / np.linalg.norm(video_embeddings_matrix, axis=1, keepdims=True)
    query_embedding_norm = query_embedding / np.linalg.norm(query_embedding)
    
    similarities = np.dot(video_embeddings_norm, query_embedding_norm)
    
    # Get top k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # Return results with metadata
    results = []
    for idx in top_indices:
        results.append({
            'frame_name': vector_metadata[idx]['frame_name'],
            'vector_index': vector_metadata[idx]['vector_index'],
            'similarity': similarities[idx],
            'distance': 1.0 - similarities[idx]  # Convert similarity to distance
        })
    
    return results

def save_index_mapping(mapping, filepath):
    """
    Saves the index mapping to a JSON file for later use.
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)
        print(f"Index mapping saved to: {filepath}")
    except Exception as e:
        print(f"Error saving index mapping: {e}")

def load_index_mapping(filepath):
    """
    Loads the index mapping from a JSON file.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        print(f"Index mapping loaded from: {filepath}")
        return mapping
    except Exception as e:
        print(f"Error loading index mapping: {e}")
        return {}

def get_vector_info_from_mapping(mapping, faiss_index):
    """
    Given a FAISS result index, returns the corresponding video_id, frame_name, and vector_index.
    """
    if faiss_index in mapping:
        info = mapping[faiss_index]
        return info['video_id'], info['frame_name'], info['vector_index']
    return None, None, None