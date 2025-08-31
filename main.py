import numpy as np
import os
import json
import torch
import clip
import pandas as pd
from modules.utils.search_utils import encode_queries, build_faiss_index, search_top_k, load_video_metadata, load_keyframes_map, load_object_detections, create_index_to_path_mapping, process_query_text

if __name__ == "__main__":
    embeddings_folder = "data/embeddings"
    queries_file_path = "queries.json"
    # frames_folder = "data/video_frames"
    
    # Khởi tạo các đường dẫn đến các thư mục dữ liệu
    media_info_folder = "data/media_info"
    map_keyframes_folder = "data/map_keyframes"
    objects_folder = "data/objects"

    # video_metadata = load_video_metadata(os.path.join(media_info_folder, "L21_V001.json"))
    # keyframes_map = load_keyframes_map(os.path.join(map_keyframes_folder, "L21_V001.csv"))
    # object_detections = load_object_detections(os.path.join(objects_folder, "001.json"))

    with open(queries_file_path, 'r', encoding='utf-8') as f:
        queries = json.load(f)

    faiss_index, _, id_mapping = build_faiss_index(embeddings_folder)
    index_to_path_mapping = create_index_to_path_mapping(embeddings_folder)
    
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")
    model, _ = clip.load("ViT-B/32", device=device)

    if faiss_index is None:
        print("Không thể tiếp tục. Vui lòng kiểm tra thư mục embeddings.")
    else:
        for query in queries:
            query_id = query.get("query_id")
            task_type = query.get("task_type")
            
            print("-" * 50)
            print(f"Bắt đầu xử lý truy vấn: {query_id} (Tác vụ: {task_type})")
            
            if task_type == "kis":
                description = query.get("description")
                query_embedding = encode_queries(description, model, device)
                distances, indices = search_top_k(query_embedding, faiss_index, k=5)

                print(indices)
                
                print("Kết quả tìm kiếm cho Textual-KVS:")
                for i, idx in enumerate(indices[0]):
                    path = id_mapping[idx]
                    distance = distances[0][i]
                    print(f"  - Top {i+1}: {path}, Khoảng cách: {distance:.4f}")
                    # if result_info:
                        # video_id = result_info["video_id"]
                        # frame_name = result_info["frame_name"]
                        # frame_path = os.path.join(frames_folder, video_id, f"{frame_name}.jpg")
                        # distance = distances[0][i]
                        # print(f"  - Top {i+1}: Video '{video_id}', Frame: '{frame_name}'")
                        # print(f"    Đường dẫn: {frame_path}")
                        # print(f"    Khoảng cách: {distance:.4f}")

            # elif task_type == "qa":
            #     # Ví dụ xử lý cho QA
            #     description = query.get("description")
            #     question = query.get("sub_queries")[-1].get("question")
            #     print(f"  - Mô tả: {description}\n  - Câu hỏi: {question}")
            #     pass # Chỗ này để bạn thêm logic xử lý cho tác vụ QA

            # elif task_type == "trake":
            #     # Ví dụ xử lý cho TRAKE
            #     description = query.get("description")
            #     print(f"  - Mô tả: {description}")
            #     for sub_query in query.get("sub_queries"):
            #         request = sub_query.get("request")
            #         print(f"  - Yêu cầu: {request}")
            #     pass # Chỗ này để bạn thêm logic xử lý cho tác vụ TRAKE

            # else:
            #     print(f"  - Tác vụ '{task_type}' không được hỗ trợ.")