import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import faiss
from modules.model.model import Clip4ClipHF

# --- Config ---
BATCH_SIZE = 1 # Adjust based on your GPU/CPU's VRAM

def process_batch(model, frame_paths):
    """Processes a batch of frames and returns their embeddings."""
    images = []
    # 1. Load all images in the batch
    for frame_path in frame_paths:
        try:
            image = Image.open(frame_path).convert("RGB")
            images.append(image)
        except Exception as e:
            print(f"Warning: Could not load image {frame_path}. Skipping. Error: {e}")
            
    if not images:
        return None, []

    # 2. Preprocess the entire batch at once
    inputs = model.processor(images=images, return_tensors="pt").to(model.device)

    # 3. Get embeddings with torch.no_grad() to save memory
    with torch.no_grad():
        embeddings = model.clip.get_image_features(**inputs)
        # Normalize embeddings
        embeddings /= embeddings.norm(dim=-1, keepdim=True)

    return embeddings.cpu().detach().numpy(), [path for path, img in zip(frame_paths, images) if img]


def generate_embeddings_batched(model, frames_folder, embeddings_folder):
    """
    Processes frames in batches to generate embeddings efficiently.
    """
    os.makedirs(embeddings_folder, exist_ok=True)

    for lot_folder in tqdm(os.listdir(frames_folder), desc="Processing Lots"):
        lot_path = os.path.join(frames_folder, lot_folder)
        if not os.path.isdir(lot_path):
            continue

        for video_id in tqdm(os.listdir(lot_path), desc=f"Processing Videos in {lot_folder}", leave=False):
            video_frame_path = os.path.join(lot_path, video_id)
            video_embedding_path = os.path.join(embeddings_folder, video_id)

            if not os.path.isdir(video_frame_path):
                continue
            os.makedirs(video_embedding_path, exist_ok=True)

            # --- Batching Logic ---
            frame_files_to_process = []
            for frame_file in os.listdir(video_frame_path):
                # Check if the output file already exists
                output_path = os.path.join(video_embedding_path, f"{os.path.splitext(frame_file)[0]}.npy")
                if not os.path.exists(output_path) and frame_file.lower().endswith((".jpg", ".png")):
                    frame_files_to_process.append(os.path.join(video_frame_path, frame_file))
            
            # Process the collected frames in batches
            for i in range(0, len(frame_files_to_process), BATCH_SIZE):
                batch_paths = frame_files_to_process[i:i + BATCH_SIZE]
                
                try:
                    embeddings, processed_paths = process_batch(model, batch_paths)
                    
                    if embeddings is not None:
                        # Save each embedding from the batch
                        for j, embedding in enumerate(embeddings):
                            original_path = processed_paths[j]
                            frame_basename = os.path.basename(original_path)
                            output_filename = f"{os.path.splitext(frame_basename)[0]}.npy"
                            output_path = os.path.join(video_embedding_path, output_filename)
                            np.save(output_path, embedding)

                except Exception as e:
                    print(f"Error processing batch starting with {batch_paths[0]}: {e}")
                    # Optional: Clear CUDA cache in case of a severe error
                    # if torch.cuda.is_available():
                    #     torch.cuda.empty_cache()


if __name__ == "__main__":
    frames_folder = "data/video_frames" 
    embeddings_folder = "data/embeddings"

    print("Loading CLIP model...")
    # Ensure model is on the correct device (GPU is highly recommended)
    clip_model = Clip4ClipHF()
    
    generate_embeddings_batched(clip_model, frames_folder, embeddings_folder)
    print("Embeddings generated successfully.")