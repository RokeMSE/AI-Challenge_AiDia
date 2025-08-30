# main.py
import numpy as np
import os
import json
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
from collections import defaultdict

from modules.model.model import Clip4ClipHF
from modules.utils.search_utils import build_faiss_index, search_top_k, create_index_to_path_mapping, find_best_frame_in_video

if __name__ == "__main__":
    # --- CONFIGURATION ---
    # Point these to your main data folders
    embeddings_folder = "data/embeddings/L21_V001"
    frames_folder = "data/video_frames"
    queries_file_path = "queries.json"
    
    # --- MODEL INITIALIZATION ---
    print("Loading CLIP model via Hugging Face...")
    model = Clip4ClipHF(model_name="openai/clip-vit-base-patch32")
    
    print("Loading VQA model (BLIP)...")
    vqa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    vqa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(model.device)

    # --- INDEXING ---
    print("Building FAISS index from all embedding folders...")
    faiss_index = build_faiss_index(embeddings_folder)
    index_to_path_mapping = create_index_to_path_mapping(embeddings_folder)

    # --- QUERY PROCESSING ---
    with open(queries_file_path, 'r', encoding='utf-8') as f:
        queries = json.load(f)

    if faiss_index is None:
        print("Could not build FAISS index. Exiting. Please run build_index.py first.")
    else:
        for query in queries:
            query_id = query.get("query_id")
            task_type = query.get("task_type")
            description = query.get("description")
            
            print("-" * 50)
            print(f"Processing query: {query_id} (Task: {task_type})")
            
            if task_type == "kis":
                query_embedding = model.get_text_features([description])
                distances, indices = search_top_k(query_embedding, faiss_index, k=5)

                print("Top 5 search results for KIS:")
                for i, idx in enumerate(indices[0]):
                    result_info = index_to_path_mapping.get(idx)
                    if result_info:
                        print(f"  - Rank {i+1}: Video '{result_info['video_id']}', Frame: '{result_info['frame_name']}' (Distance: {distances[0][i]:.4f})")

            elif task_type == "trake":
                # Stage 1: Retrieve a larger pool of candidate frames for the main description
                query_embedding = model.get_text_features([description])
                _, candidate_indices = search_top_k(query_embedding, faiss_index, k=100) # Get top 100 candidates

                # Group candidates by video ID to find the most likely video
                video_scores = defaultdict(int)
                for idx in candidate_indices[0]:
                    video_id = index_to_path_mapping.get(idx, {}).get("video_id")
                    if video_id:
                        video_scores[video_id] += 1
                
                if not video_scores:
                    print("  - No matching videos found for the main description.")
                    continue

                # The video with the most top-ranking frames is our candidate
                candidate_video_id = max(video_scores, key=video_scores.get)
                print(f"  - Best candidate video identified: {candidate_video_id}")

                # Stage 2: Find the best frame within that video for each sub-query
                predicted_frames = []
                for sub_query in query.get("sub_queries"):
                    step_request = sub_query.get("request")
                    best_frame, score = find_best_frame_in_video(step_request, candidate_video_id, embeddings_folder, model)
                    if best_frame:
                        predicted_frames.append(best_frame)
                    else:
                        predicted_frames.append("0") # Use "0" as a placeholder if no frame is found
                        
                submission_string = f"{candidate_video_id}, " + ", ".join(predicted_frames)
                print(f"  - Final TRAKE Submission: {submission_string}")