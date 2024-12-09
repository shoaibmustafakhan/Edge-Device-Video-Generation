# inference.py
import torch
import cv2
import numpy as np
from config import *
from model import ConvLSTM
import matplotlib.pyplot as plt

def load_model_for_inference():
    model = ConvLSTM().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "checkpoint.pth")
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    else:
        print("No checkpoint found. Using untrained model.")
    
    model.eval()
    return model

def process_video_for_prediction(video_path, num_input_frames=INPUT_TIMESTEPS):
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    for _ in range(num_input_frames):
        ret, frame = cap.read()
        if not ret:
            cap.release()
            raise ValueError("Video is too short!")
            
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized_frame = cv2.resize(gray_frame, IMG_SIZE)
        normalized_frame = resized_frame.astype(np.float32) / 255.0
        frames.append(normalized_frame)
    
    actual_next_frames = []
    for _ in range(OUTPUT_TIMESTEPS):
        ret, frame = cap.read()
        if ret:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized_frame = cv2.resize(gray_frame, IMG_SIZE)
            normalized_frame = resized_frame.astype(np.float32) / 255.0
            actual_next_frames.append(normalized_frame)
    
    cap.release()
    return np.array(frames), np.array(actual_next_frames) if actual_next_frames else None

def save_frames_as_video(frames, filename, fps=30):
    frames_uint8 = (frames * 255).astype(np.uint8)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, IMG_SIZE, isColor=False)
    
    for frame in frames_uint8:
        out.write(frame)
    
    out.release()
    print(f"Saved video to {filename}")

def save_comparison_video(predicted_frames, actual_frames, filename="comparison.mp4"):
    """Creates a side-by-side comparison video without text labels"""
    predicted_uint8 = (predicted_frames * 255).astype(np.uint8)
    actual_uint8 = (actual_frames * 255).astype(np.uint8)
    
    h, w = IMG_SIZE
    combined_size = (w*2, h)  # Double width for side by side
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, 30, combined_size, isColor=False)
    
    for pred_frame, act_frame in zip(predicted_uint8, actual_uint8):
        combined = np.hstack([pred_frame, act_frame])
        out.write(combined)
    
    out.release()
    print(f"Saved comparison video to {filename}")

def predict_next_frames(video_path):
    """Predict next frames given a video file."""
    # Load model
    model = load_model_for_inference()
    
    # Process video
    input_frames, actual_frames = process_video_for_prediction(video_path)
    
    # Prepare input for model
    input_tensor = torch.tensor(input_frames).unsqueeze(0).unsqueeze(0).to(DEVICE)
    
    # Get prediction
    with torch.no_grad():
        predicted_frames = model(input_tensor)
        predicted_frames = predicted_frames.cpu().numpy()[0, 0]
    
    # Save videos
    save_frames_as_video(predicted_frames, "predicted_sequence.mp4")
    if actual_frames is not None:
        save_frames_as_video(actual_frames, "actual_sequence.mp4")
        save_comparison_video(predicted_frames, actual_frames)
    
    # Display frames
    plt.figure(figsize=(20, 8))
    
    # Plot input sequence
    for i in range(INPUT_TIMESTEPS):
        plt.subplot(3, max(INPUT_TIMESTEPS, OUTPUT_TIMESTEPS), i + 1)
        plt.imshow(input_frames[i], cmap='gray')
        plt.title(f'Input {i+1}')
        plt.axis('off')
    
    # Plot predicted sequence
    for i in range(OUTPUT_TIMESTEPS):
        plt.subplot(3, max(INPUT_TIMESTEPS, OUTPUT_TIMESTEPS), i + 1 + max(INPUT_TIMESTEPS, OUTPUT_TIMESTEPS))
        plt.imshow(predicted_frames[i], cmap='gray')
        plt.title(f'Predicted {i+1}')
        plt.axis('off')
    
    # Plot actual sequence if available
    if actual_frames is not None:
        for i in range(len(actual_frames)):
            plt.subplot(3, max(INPUT_TIMESTEPS, OUTPUT_TIMESTEPS), i + 1 + 2*max(INPUT_TIMESTEPS, OUTPUT_TIMESTEPS))
            plt.imshow(actual_frames[i], cmap='gray')
            plt.title(f'Actual {i+1}')
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return predicted_frames, actual_frames

# Specify your video path here
if __name__ == "__main__":
    VIDEO_PATH = r"C:\University\ConvLSTM\train\Biking\v_Biking_g03_c04.avi"  # Replace with your video path
    predicted_frames, actual_frames = predict_next_frames(VIDEO_PATH)