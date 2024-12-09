import os
import cv2
import numpy as np
from tqdm import tqdm

def preprocess_dataset(data_root, output_dir, size_limit_kb=4000, classes=["WalkingWithDog", "Biking", "JumpingJack", "Bowling"]):
    os.makedirs(output_dir, exist_ok=True)
    
    for class_name in classes:
        print(f"Processing {class_name}")
        class_path = os.path.join(data_root, class_name)
        output_class_dir = os.path.join(output_dir, class_name)
        os.makedirs(output_class_dir, exist_ok=True)
        
        videos = [f for f in os.listdir(class_path) if f.endswith('.avi')]
        for video in tqdm(videos):
            video_path = os.path.join(class_path, video)
            output_path = os.path.join(output_class_dir, video.replace('.avi', '.npy'))
            
            if os.path.exists(output_path) and os.path.getsize(output_path) / 1024 <= size_limit_kb:
                continue
                
            cap = cv2.VideoCapture(video_path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (64, 64))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(frame)
            cap.release()
            
            if frames:
                frames = np.array(frames) / 255.0
                # Truncate frames if numpy file would be too large
                estimated_size = frames.nbytes / 1024
                if estimated_size > size_limit_kb:
                    max_frames = int((size_limit_kb * 1024) / (64 * 64))
                    frames = frames[:max_frames]
                
                np.save(output_path, frames)
                
                # Verify size
                if os.path.getsize(output_path) / 1024 > size_limit_kb:
                    os.remove(output_path)
                    print(f"Skipping {video} - too large even after truncation")

if __name__ == "__main__":
    preprocess_dataset("train", "processed_data")