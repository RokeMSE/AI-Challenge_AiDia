import numpy as np
import os
import json
import torch
import clip
import pandas as pd
from modules.utils.search_utils import build_faiss_index, search_top_k

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

if __name__ == "__main__":
    embeddings_folder = "data/embeddings"
    queries_file_path = "queries.json"
    frames_folder = "data/video_frames"
    
    # Khởi tạo các đường dẫn đến các thư mục dữ liệu
    media_info_folder = "data/media_info"
    map_keyframes_folder = "data/map_keyframes"
    objects_folder = "data/objects"
    
    with open(queries_file_path, 'r', encoding='utf-8') as f:
        queries = json.load(f)

    faiss_index, _ = build_faiss_index(embeddings_folder)
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
                if query_id == "p1-1-kis":
                    sub_queries = [
                        "Cảnh quay bằng flycam một cây cầu ở TP Hồ Chí Minh.",
                        "cảnh quay tòa nhà Bitexco.",
                        "quay hình ảnh hồ gươm tại Hà Nội."
                    ]
                else:
                    sub_queries = [description]

                all_query_embeddings = [process_query_text(sub_q, model, device) for sub_q in sub_queries]
                query_embedding = np.mean(all_query_embeddings, axis=0)
                distances, indices = search_top_k(query_embedding, faiss_index, k=5)

                print("Kết quả tìm kiếm cho Textual-KVS:")
                for i, idx in enumerate(indices[0]):
                    result_info = index_to_path_mapping.get(idx)
                    print(result_info)
                    if result_info:
                        print("Khôi")
                        video_id = result_info["video_id"]
                        frame_name = result_info["frame_name"]
                        frame_path = os.path.join(frames_folder, video_id, f"{frame_name}.jpg")
                        distance = distances[0][i]
                        print(f"  - Top {i+1}: Video '{video_id}', Frame: '{frame_name}'")
                        print(f"    Đường dẫn: {frame_path}")
                        print(f"    Khoảng cách: {distance:.4f}")
            
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