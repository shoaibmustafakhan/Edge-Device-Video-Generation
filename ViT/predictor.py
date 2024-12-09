import os
import cv2
import numpy as np
import tensorflow as tf

def process_and_predict_video(
    input_video_path,
    model,
    output_path="predicted_video.mp4",
    frame_size=(64, 64),
    input_frames=10,
    future_frames=10,
    fps=30
):
    """
    Process a video file and generate future frame predictions
    
    Args:
        input_video_path: Path to input .avi video
        model: Loaded ViViT model
        output_path: Path to save predicted video
        frame_size: Tuple of (height, width) for frame resizing
        input_frames: Number of input frames to use
        future_frames: Number of frames to predict
        fps: Frames per second for output video
    """
    # Process video if it's an .avi file
    if input_video_path.endswith('.avi'):
        video_capture = cv2.VideoCapture(input_video_path)
        frames = []
        
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
                
            frame = cv2.resize(frame, frame_size)
            frames.append(frame)
            
        video_capture.release()
        frames = np.array(frames)
        
    # Load directly if it's already a .npy file
    elif input_video_path.endswith('.npy'):
        frames = np.load(input_video_path)
    else:
        raise ValueError("Input video must be .avi or .npy format")

    # Convert to grayscale and normalize
    frames = np.mean(frames, axis=-1, keepdims=True)
    frames = frames.astype(np.float32) / 255.0

    # Ensure we have enough frames
    if len(frames) < input_frames:
        raise ValueError(f"Video must have at least {input_frames} frames")

    # Get initial sequence
    input_sequence = frames[:input_frames]
    input_sequence = np.expand_dims(input_sequence, 0)  # Add batch dimension

    # Generate predictions
    predicted_frames = model.predict(input_sequence)

    # Save the predicted frames as video
    if output_path:
        predicted_frames = predicted_frames[0]  # Remove batch dimension
        predicted_frames = (predicted_frames * 255).astype(np.uint8)
        
        height, width = frame_size
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)
        
        try:
            for frame in predicted_frames:
                frame = np.squeeze(frame, axis=-1)  # Remove channel dimension
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                out.write(frame_bgr)
        finally:
            out.release()

    return predicted_frames

# Example usage:
"""
model = load_model_for_inference()
predictions = process_and_predict_video(
    "path/to/video.avi",
    model,
    "output_prediction.mp4"
)
"""