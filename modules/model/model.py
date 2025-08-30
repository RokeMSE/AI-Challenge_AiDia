# modules/model/model.py
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from decord import VideoReader, cpu
import numpy as np

class Clip4ClipHF(nn.Module):
    """
    A Hugging Face-based wrapper for CLIP4Clip.
    It uses the official CLIPModel and adapts it for video processing.
    """
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        super().__init__()
        # Load the core CLIP model and its processor from Hugging Face
        self.clip = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)
        print(f"Model '{model_name}' loaded on device: {self.device}")

    def get_text_features(self, text_queries: list) -> np.ndarray:
        with torch.no_grad():
            inputs = self.processor(text=text_queries, return_tensors="pt", padding=True, truncation=True) # Process text inputs
            text_features = self.clip.get_text_features(**inputs.to(self.device)) # Get text features
            text_features /= text_features.norm(dim=-1, keepdim=True) # Normalize
        return text_features.cpu().numpy().astype('float32') # Convert to numpy array

    def get_video_features(self, video_path: str, num_frames: int = 16) -> np.ndarray: # num_frames: change this to the actual number of frames
        with torch.no_grad():
            # 1. Read video and sample frames using decord
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            frame_indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
            frames = vr.get_batch(frame_indices).asnumpy()

            # 2. Process frames with the CLIPProcessor
            inputs = self.processor(images=list(frames), return_tensors="pt")
            pixel_values = inputs['pixel_values'].to(self.device)
            
            # 3. Get frame features from the model
            frame_features = self.clip.get_image_features(pixel_values=pixel_values)
            
            # 4. Aggregate into one feature
            video_feature = frame_features.mean(dim=0) # Mean pooling
            video_feature /= video_feature.norm(dim=-1, keepdim=True) # Normalize

        return video_feature.cpu().numpy().astype('float32')