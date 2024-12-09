import os
import numpy as np
import tensorflow as tf
import cv2
from model_architecture import ViViT
from training_evaluation import predict_future_frames, visualize_predictions

def save_frames_as_video(frames, output_path, fps=30):
    """
    Save a sequence of frames as a video file.
    Handles grayscale frames correctly and ensures proper format conversion.
    """
    # Convert frames to uint8 if they're float
    if frames.dtype == np.float32 or frames.dtype == np.float64:
        frames = (frames * 255).astype(np.uint8)
    
    # Get dimensions and ensure proper shape
    if len(frames.shape) == 4 and frames.shape[-1] == 1:
        frames = np.squeeze(frames, axis=-1)  # Remove single channel dimension
    
    height, width = frames.shape[1:] if len(frames.shape) == 3 else frames.shape

    # Initialize video writer for grayscale video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)
    
    try:
        for frame in frames:
            # Convert grayscale to BGR (required by OpenCV)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            out.write(frame_bgr)
            
        print(f"Successfully wrote {len(frames)} frames to {output_path}")
    except Exception as e:
        print(f"Error saving video: {e}")
    finally:
        out.release()

def load_model_for_inference(weights_path="vivit_grayscale_model.keras"):
    # Create model with same architecture
    model = ViViT(
        input_shape=(10, 64, 64, 1),
        patch_size=8,
        embed_dim=256,
        num_heads=16,
        ff_dim=512,
        num_transformer_layers=8,
        dropout=0.1,
        future_frames=10
    )
    
    # Build the model first with a dummy input
    dummy_input = tf.zeros((1, 10, 64, 64, 1))
    _ = model(dummy_input)
    
    # Compile the model (needed for loading weights)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='mse'  # Doesn't matter for inference, just needed for loading
    )
    
    # Load weights
    try:
        model.load_weights(weights_path)
        print(f"Successfully loaded weights from {weights_path}")
    except Exception as e:
        print(f"Error loading weights: {e}")
        raise e
    
    return model

def run_inference(video_path, output_path="predicted_video.mp4"):
    # Load model
    print("Loading model...")
    model = load_model_for_inference()
    
    # Load and preprocess video
    print(f"Processing video from {video_path}")
    test_video = np.load(video_path)
    test_video = np.mean(test_video, axis=-1, keepdims=True)
    input_frames = test_video[:10]
    
    # Get predictions
    print("Generating predictions...")
    predicted_frames = predict_future_frames(model, input_frames)
    
    # Visualize results
    print("Visualizing results...")
    visualize_predictions(input_frames, predicted_frames[0])
    
    # Save as video
    print(f"Saving video to {output_path}")
    save_frames_as_video(predicted_frames[0], output_path)
    print(f"Done! Prediction video saved as '{output_path}'")

if __name__ == "__main__":
    # Example usage
    video_path = "C:/University/DeepLearning/processed_data/test/JumpingJack/v_JumpingJack_g01_c03.npy"  # Change this to your video path
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        print("Please provide the correct path to your test video")
        exit(1)
    
    run_inference(video_path)