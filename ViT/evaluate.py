import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from model_architecture import ViViT

def predict_future_frames(model, input_sequence, device='cuda'):
    model.eval()
    with torch.no_grad():
        input_sequence = torch.from_numpy(input_sequence).float() / 255.0
        input_sequence = input_sequence.permute(3, 0, 1, 2).unsqueeze(0).to(device)
        predicted_frames = model(input_sequence)
        predicted_frames = predicted_frames.cpu().numpy()
        predicted_frames = np.clip(predicted_frames * 255, 0, 255).astype(np.uint8)
        predicted_frames = predicted_frames.transpose(0, 2, 3, 4, 1)
    return predicted_frames

def visualize_predictions(input_frames, predicted_frames):
    num_input_frames = input_frames.shape[0]
    num_predicted_frames = predicted_frames.shape[0]

    fig, axes = plt.subplots(2, max(num_input_frames, num_predicted_frames), figsize=(15, 5))
    
    for i in range(num_input_frames):
        axes[0, i].imshow(input_frames[i, :, :, 0], cmap="gray")
        axes[0, i].axis("off")
        axes[0, i].set_title(f"Input Frame {i+1}")

    for i in range(num_predicted_frames):
        axes[1, i].imshow(predicted_frames[i, :, :, 0], cmap="gray")
        axes[1, i].axis("off")
        axes[1, i].set_title(f"Predicted Frame {i+1}")

    plt.tight_layout()
    plt.show()

def save_frames_as_video(frames, output_path, fps=30):
    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(output_path, 
                         cv2.VideoWriter_fourcc(*'mp4v'), 
                         fps, 
                         (width, height), 
                         isColor=False)
    
    for frame in frames:
        frame = np.squeeze(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        out.write(frame)
    
    out.release()

def evaluate_model(model, test_path, device='cuda'):
    # Create output directory
    output_dir = 'evaluation_results'
    os.makedirs(output_dir, exist_ok=True)

    # Process test videos
    all_metrics = []
    
    # Example test video
    test_video_path = os.path.join(test_path, "Bowling/v_Bowling_g01_c01.npy")
    test_video = np.load(test_video_path)
    test_video = np.mean(test_video, axis=-1, keepdims=True)
    
    # Get input frames
    input_frames = test_video[:10]
    
    # Predict future frames
    predicted_frames = predict_future_frames(model, input_frames, device)
    
    # Visualize results
    save_path = os.path.join(output_dir, 'prediction.png')
    visualize_predictions(input_frames, predicted_frames[0])
    
    # Save as video
    video_path = os.path.join(output_dir, 'prediction.mp4')
    save_frames_as_video(predicted_frames[0], video_path)
    
    return predicted_frames