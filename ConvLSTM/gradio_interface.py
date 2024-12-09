# gradio_app.py
import gradio as gr
import torch
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error
from config import *
from model import ConvLSTM
import os
import tempfile

def load_model():
    model = ConvLSTM().to(DEVICE)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "checkpoint.pth")
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    else:
        print("No checkpoint found. Using untrained model.")
    
    model.eval()
    return model

def process_video(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    for _ in range(INPUT_TIMESTEPS):
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

def save_video(frames, path, fps=30):
    frames_uint8 = (frames * 255).astype(np.uint8)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, IMG_SIZE, isColor=False)
    
    for frame in frames_uint8:
        out.write(frame)
    out.release()

def save_comparison_video(predicted_frames, actual_frames, path):
    predicted_uint8 = (predicted_frames * 255).astype(np.uint8)
    actual_uint8 = (actual_frames * 255).astype(np.uint8)
    
    h, w = IMG_SIZE
    combined_size = (w*2, h)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, 30, combined_size, isColor=False)
    
    for pred_frame, act_frame in zip(predicted_uint8, actual_uint8):
        combined = np.hstack([pred_frame, act_frame])
        out.write(combined)
    
    out.release()

def calculate_metrics(predicted_frames, actual_frames):
    # Calculate SSIM
    ssim_scores = []
    for pred, actual in zip(predicted_frames, actual_frames):
        score = ssim(pred, actual, data_range=1.0)
        ssim_scores.append(score)
    avg_ssim = np.mean(ssim_scores)
    
    # Calculate MSE
    mse = mean_squared_error(actual_frames.flatten(), predicted_frames.flatten())
    
    return avg_ssim, mse

def predict_video(video_path):
    try:
        # Create temporary files for the videos
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_pred, \
             tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_comp:
            
            pred_path = temp_pred.name
            comp_path = temp_comp.name
            
            # Process input video
            input_frames, actual_frames = process_video(video_path)
            
            # Prepare input for model
            input_tensor = torch.tensor(input_frames).unsqueeze(0).unsqueeze(0).to(DEVICE)
            
            # Get prediction
            with torch.no_grad():
                predicted_frames = model(input_tensor)
                predicted_frames = predicted_frames.cpu().numpy()[0, 0]
            
            # Calculate metrics
            avg_ssim, mse = calculate_metrics(predicted_frames, actual_frames)
            metrics_text = f"Average SSIM: {avg_ssim:.4f}\nMSE: {mse:.4f}"
            
            # Save videos
            save_video(predicted_frames, pred_path)
            save_comparison_video(predicted_frames, actual_frames, comp_path)
            
            return [pred_path, comp_path, metrics_text]
            
    except Exception as e:
        return [None, None, f"Error: {str(e)}"]

# Load model globally
model = load_model()

# Create Gradio interface
iface = gr.Interface(
    fn=predict_video,
    inputs=[
        gr.Video(label="Upload Video")
    ],
    outputs=[
        gr.Video(label="Predicted Sequence"),
        gr.Video(label="Side-by-Side Comparison"),
        gr.Textbox(label="Metrics")
    ],
    title="Video Frame Prediction",
    description="Upload a video to predict the next 10 frames. The model will show the predicted sequence and a side-by-side comparison with actual frames."
)

if __name__ == "__main__":
    iface.launch(share=True)