import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import faiss
from modules.model.model import Clip4ClipHF
from modules.utils.search_utils import create_index_to_path_mapping

def generate_embeddings(model, frames_folder, embeddings_folder):
    os.makedirs(embeddings_folder, exist_ok=True)
    for video_id in tqdm(os.listdir(frames_folder), desc="Processing Videos"):
        video_frame_path = os.path.join(frames_folder, video_id)
        video_embedding_path = os.path.join(embeddings_folder, video_id)
        if not os.path.isdir(video_frame_path): continue
        os.makedirs(video_embedding_path, exist_ok=True)

        for frame_file in os.listdir(video_frame_path):
            if frame_file.lower().endswith((".jpg", ".png")):
                frame_path = os.path.join(video_frame_path, frame_file)
                output_path = os.path.join(video_embedding_path, f"{os.path.splitext(frame_file)[0]}.npy")
                if os.path.exists(output_path): continue
                try:
                    image = Image.open(frame_path).convert("RGB")
                    inputs = model.processor(images=[image], return_tensors="pt").to(model.device)
                    embedding = model.clip.get_image_features(**inputs)
                    embedding /= embedding.norm(dim=-1, keepdim=True)
                    np.save(output_path, embedding.cpu().detach().numpy())
                except Exception as e:
                    print(f"Error processing {frame_path}: {e}")

if __name__ == "__main__":
    frames_folder = "data/video_frames/Keyframes_L21"
    embeddings_folder = "data/embeddings"

    print("Loading CLIP model...")
    clip_model = Clip4ClipHF()
    
    generate_embeddings(clip_model, frames_folder, embeddings_folder)
    
    print("\nEmbeddings generated. You can now run main.py to search.")