# dataset.py
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from config import *

class VideoDataset(Dataset):
    def __init__(self, video_paths, input_timesteps, output_timesteps, img_size):
        self.video_paths = video_paths
        self.input_timesteps = input_timesteps
        self.output_timesteps = output_timesteps
        self.img_size = img_size
        self.sequence_info = self._index_sequences()

    def _index_sequences(self):
        sequence_info = []
        
        for video_idx, video_path in enumerate(self.video_paths):
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            # Only consider sequences if we have enough frames for both input and output
            required_frames = self.input_timesteps + self.output_timesteps
            if total_frames >= required_frames:
                # Calculate how many complete sequences we can get from this video
                num_sequences = total_frames - required_frames + 1
                
                for seq_start in range(num_sequences):
                    sequence_info.append({
                        'video_idx': video_idx,
                        'start_frame': seq_start
                    })
        
        return sequence_info

    def __len__(self):
        return len(self.sequence_info)

    def __getitem__(self, idx):
        info = self.sequence_info[idx]
        video_path = self.video_paths[info['video_idx']]
        start_frame = info['start_frame']
        
        frames = []
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Read exactly input_timesteps + output_timesteps frames
        required_frames = self.input_timesteps + self.output_timesteps
        for _ in range(required_frames):
            ret, frame = cap.read()
            if not ret:  # Should never happen due to our _index_sequences check
                # Fill with zeros if we somehow got here
                missing_frames = required_frames - len(frames)
                for _ in range(missing_frames):
                    frames.append(np.zeros(self.img_size, dtype=np.float32))
                break
                
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized_frame = cv2.resize(gray_frame, self.img_size)
            normalized_frame = resized_frame.astype(np.float32) / 255.0
            frames.append(normalized_frame)
        
        cap.release()
        
        # Split into input and target sequences
        frames = np.array(frames)
        input_sequence = frames[:self.input_timesteps]
        target_sequence = frames[self.input_timesteps:required_frames]
        
        # Add channel dimension
        input_sequence = np.expand_dims(input_sequence, axis=0)
        target_sequence = np.expand_dims(target_sequence, axis=0)
        
        return torch.tensor(input_sequence, dtype=torch.float32), torch.tensor(target_sequence, dtype=torch.float32)

def check_video_file(video_path, max_size_kb=4000):
    """Check if video file meets the size criteria."""
    file_size_kb = os.path.getsize(video_path) / 1024
    if file_size_kb > max_size_kb:
        return False
    
    # Also check if video has enough frames
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    return total_frames >= (INPUT_TIMESTEPS + OUTPUT_TIMESTEPS)

def get_valid_video_paths(data_dir, max_size_kb=4000):
    """Get paths of all valid videos."""
    valid_paths = []
    skipped_size = 0
    skipped_frames = 0
    
    classes = [cls for cls in SELECTED_CLASSES if os.path.exists(os.path.join(data_dir, cls))]
    
    for cls in classes:
        class_path = os.path.join(data_dir, cls)
        video_files = [f for f in os.listdir(class_path) 
                      if f.endswith(('.avi', '.mp4', '.mov'))]
        
        for video_file in video_files:
            video_path = os.path.join(class_path, video_file)
            file_size_kb = os.path.getsize(video_path) / 1024
            
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            if file_size_kb > max_size_kb:
                skipped_size += 1
                continue
            
            if total_frames < (INPUT_TIMESTEPS + OUTPUT_TIMESTEPS):
                skipped_frames += 1
                continue
                
            valid_paths.append(video_path)
    
    print(f"Skipped {skipped_size} videos due to size > {max_size_kb}KB")
    print(f"Skipped {skipped_frames} videos due to insufficient frames")
    
    return valid_paths

def get_dataloaders(data_dir, batch_size=BATCH_SIZE, max_size_kb=4000):
    print("Finding valid videos...")
    video_paths = get_valid_video_paths(data_dir, max_size_kb)
    print(f"Found {len(video_paths)} valid videos")
    
    if not video_paths:
        raise ValueError("No valid videos found!")
    
    # Split video paths into train and validation
    train_paths, val_paths = train_test_split(video_paths, test_size=0.2, random_state=42)
    
    # Create datasets
    train_dataset = VideoDataset(train_paths, INPUT_TIMESTEPS, OUTPUT_TIMESTEPS, IMG_SIZE)
    val_dataset = VideoDataset(val_paths, INPUT_TIMESTEPS, OUTPUT_TIMESTEPS, IMG_SIZE)
    
    print(f"Training sequences: {len(train_dataset)}")
    print(f"Validation sequences: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=2, 
                            pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                          shuffle=False, num_workers=2, 
                          pin_memory=True)
    
    return train_loader, val_loader