# main.py
import numpy as np
import os
import json
import torch
import pandas as pd

from modules.model.model import Clip4ClipHF # from modules/model/model.py
from modules.utils.search_utils import *

if __name__ == "__main__":
    embeddings_folder = "data/embeddings"
    queries_file_path = "queries.json"
    frames_folder = "data/video_frames"
    
    # 1. Initialize our new Hugging Face-based model
    print("Loading CLIP model via Hugging Face...")
    model = Clip4ClipHF(model_name="openai/clip-vit-base-patch32")

    # 2. Load the pre-computed FAISS index and mappings
    print("Building FAISS index from pre-computed embeddings...")
    faiss_index, _ = build_faiss_index(embeddings_folder)
    index_to_path_mapping = create_index_to_path_mapping(embeddings_folder)

    # 3. Load queries
    with open(queries_file_path, 'r', encoding='utf-8') as f:
        queries = json.load(f)

    if faiss_index is None:
        print("Could not build FAISS index. Please check the embeddings folder.")
    else:
        # 4. Process each query
        for query in queries:
            query_id = query.get("query_id")
            task_type = query.get("task_type")
            description = query.get("description")
            
            print("-" * 50)
            print(f"Processing query: {query_id} (Task: {task_type})")
            
            if task_type == "kis":
                # Use our new model to get the text embedding
                query_embedding = model.get_text_features([description])
                
                # Search the FAISS index 
                distances, indices = search_top_k(query_embedding, faiss_index, k=5)

                print("Top 5 search results for Textual-KIS:")
                for i, idx in enumerate(indices[0]):
                    result_info = index_to_path_mapping.get(idx)
                    if result_info:
                        video_id = result_info["video_id"]
                        frame_name = result_info["frame_name"]
                        frame_path = os.path.join(frames_folder, video_id, f"{frame_name}.jpg")
                        distance = distances[0][i]
                        print(f"  - Rank {i+1}: Video '{video_id}', Frame: '{frame_name}'")
                        print(f"    Path: {frame_path}")
                        print(f"    Distance: {distance:.4f}")

            elif task_type == "qa":
                pass

            elif task_type == "trake":
                description = query.get("description")
                
                # Stage 1: Video Retrieval (Simplified for this example)
                # NOTE: For the real contest, you should build and search a video-level FAISS index.
                # Here, we'll just assume a method to get the most likely video_id.
                # Let's pretend 'L10_V010' is our best guess from the video-level search.
                candidate_video_id = "L10_V010" # Replace with actual video search result
                print(f"  - Best candidate video from description: {candidate_video_id}")

                # Stage 2: Localize frame for each step
                predicted_frames = []
                for sub_query in query.get("sub_queries"):
                    step_request = sub_query.get("request")
                    print(f"    - Finding frame for request: '{step_request}'")
                    
                    best_frame, score = find_best_frame_in_video(step_request, candidate_video_id, embeddings_folder, model)
                    if best_frame:
                        predicted_frames.append(best_frame)
                        print(f"      -> Found frame: {best_frame} (Score: {score:.4f})")
                    else:
                        predicted_frames.append("0") # Placeholder if frame not found
                        
                # Compile and print the final submission string
                submission_string = f"{candidate_video_id}, " + ", ".join(predicted_frames)
                print(f"\n  - Final TRAKE Submission: {submission_string}")