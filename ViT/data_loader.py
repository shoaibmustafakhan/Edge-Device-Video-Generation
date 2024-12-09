import os
import cv2
import numpy as np
import tensorflow as tf

def preprocess_videos(output_path, file_path, selected_classes, frame_size=(64, 64), convert_to_grayscale=False):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for class_name in selected_classes:
        class_path = os.path.join(file_path, class_name)
        processed_class_path = os.path.join(output_path, class_name)

        if not os.path.exists(processed_class_path):
            os.makedirs(processed_class_path)

        # Loop through each video in the class folder
        for video_file in os.listdir(class_path):
            if video_file.endswith(".avi"):
                video_path = os.path.join(class_path, video_file)
                video_capture = cv2.VideoCapture(video_path)

                frames = []
                while True:
                    ret, frame = video_capture.read()
                    if not ret:
                        break

                    # Resize frame
                    frame = cv2.resize(frame, frame_size)

                    # Convert to grayscale if needed
                    if convert_to_grayscale:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    frames.append(frame)

                video_capture.release()

                # Save preprocessed frames as a numpy array
                frames = np.array(frames)
                save_path = os.path.join(processed_class_path, video_file.replace(".avi", ".npy"))
                np.save(save_path, frames)

                print(f"Processed and saved: {save_path}")

def prepare_dataset():
    train_path = "./train"
    val_path = "./val"
    test_path = "./test"
    
    output_train = "./processed_data/train"
    output_val = "./processed_data/val"
    output_test = "./processed_data/test"
    
    selected_classes = ["Basketball", "WalkingWithDog", "Biking", "JumpingJack", "Bowling"]
    
    print("Processing training data...")
    preprocess_videos(output_train, train_path, selected_classes)
    
    print("Processing validation data...")
    preprocess_videos(output_val, val_path, selected_classes)
    
    print("Processing test data...")
    preprocess_videos(output_test, test_path, selected_classes)
    
    return output_train, output_val, output_test

class VideoFrameDataset(tf.keras.utils.Sequence):
    def __init__(self, data_path, input_frames=5, future_frames=5, batch_size=8):
        self.data_path = data_path
        self.input_frames = input_frames
        self.future_frames = future_frames
        self.batch_size = batch_size
        self.video_files = []

        for class_name in os.listdir(data_path):
            class_path = os.path.join(data_path, class_name)
            if os.path.isdir(class_path):
                for video_file in os.listdir(class_path):
                    if video_file.endswith('.npy'):
                        self.video_files.append(os.path.join(class_path, video_file))

    def __len__(self):
        return len(self.video_files) // self.batch_size

    def __getitem__(self, idx):
        batch_files = self.video_files[idx * self.batch_size:(idx + 1) * self.batch_size]
        X, y = [], []

        for file_path in batch_files:
            video = np.load(file_path)
            if len(video) >= (self.input_frames + self.future_frames):
                video = np.mean(video, axis=-1, keepdims=True)

                start_idx = np.random.randint(0, len(video) - (self.input_frames + self.future_frames) + 1)
                input_seq = video[start_idx:start_idx + self.input_frames]
                target_seq = video[start_idx + self.input_frames:start_idx + self.input_frames + self.future_frames]

                input_seq = input_seq.astype(np.float32) / 255.0
                target_seq = target_seq.astype(np.float32) / 255.0

                X.append(input_seq)
                y.append(target_seq)

        return np.array(X), np.array(y)