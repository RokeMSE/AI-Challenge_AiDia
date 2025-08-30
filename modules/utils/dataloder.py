import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from decord import VideoReader, cpu
import numpy as np

class VideoTextDataset(Dataset):
    def __init__(self, csv_path, video_root, num_frames, processor):
        self.data = pd.read_csv(csv_path)
        self.video_root = video_root
        self.num_frames = num_frames
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_path = f"{self.video_root}/{self.data['video_id'].iloc[idx]}.mp4"
        text_caption = self.data['sentence'].iloc[idx]
        
        # Load video and sample frames
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        frames = vr.get_batch(frame_indices).asnumpy() # T, H, W, C
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2) # T, C, H, W

        # Preprocess frames with CLIP processor
        # Note: The processor expects a list of images (or a batch), so we process frame by frame
        # This is not the most efficient way, but it's clear. For speed, batch processing is better.
        processed_frames = self.processor(images=list(frames), return_tensors="pt")['pixel_values']

        return text_caption, processed_frames

def create_dataloader(config, processor):
    dataset = VideoTextDataset(
        csv_path=config['data_path'],
        video_root=config['video_root'],
        num_frames=config['num_frames'],
        processor=processor
    )
    # The collate function handles batching text and video tensors correctly
    def collate_fn(batch):
        texts, videos = zip(*batch)
        return list(texts), torch.stack(videos)

    return DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn
    )