import os
import numpy as np
import tensorflow as tf
import cv2
import gradio as gr
from model import PredRNN
from video_generator import VideoGenerator
from sklearn.metrics import mean_squared_error

def evaluate_predictions(input_frames, predictions, future_frames):
    """
    Evaluate predictions using SSIM and MSE using TensorFlow's implementation
    """
    mse_scores = []
    ssim_scores = []
    
    # Convert inputs to correct format if needed
    if len(predictions.shape) == 4:  # shape: (frames, height, width, channels)
        predictions = predictions
    if len(future_frames.shape) == 4:
        future_frames = future_frames
    
    for pred, true in zip(predictions, future_frames):
        # Calculate MSE
        mse = mean_squared_error(true.flatten(), pred.flatten())
        mse_scores.append(mse)
        
        # Calculate SSIM using TensorFlow
        # Convert to float32 explicitly
        pred_tf = tf.cast(tf.expand_dims(pred, 0), tf.float32)
        true_tf = tf.cast(tf.expand_dims(true, 0), tf.float32)
        
        ssim_value = tf.image.ssim(pred_tf, true_tf, max_val=1.0)
        ssim_scores.append(float(ssim_value.numpy()))
    
    return {
        'avg_mse': np.mean(mse_scores),
        'avg_ssim': np.mean(ssim_scores),
        'mse_per_frame': mse_scores,
        'ssim_per_frame': ssim_scores
    }

def process_video(video_path, timesteps=10):
    """Process video file into model-compatible format"""
    print(f"Processing video from path: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    
    if not cap.isOpened():
        raise ValueError("Failed to open video file")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (64, 64))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = frame / 255.0
        frames.append(frame)
        
    cap.release()
    
    if len(frames) < timesteps * 2:
        raise ValueError(f"Video needs at least {timesteps * 2} frames for evaluation! Got {len(frames)} frames")
    
    print(f"Processed {len(frames)} frames")
    
    input_frames = np.array(frames[:timesteps])
    future_frames = np.array(frames[timesteps:timesteps*2])
    
    input_frames = np.expand_dims(input_frames, axis=-1)
    future_frames = np.expand_dims(future_frames, axis=-1)
    
    return np.expand_dims(input_frames, axis=0), future_frames

def load_model_for_inference(weights_path):
    """Load the PredRNN model for inference"""
    try:
        # Initialize model
        model_builder = PredRNN()
        input_shape = (10, 64, 64, 1)
        model = model_builder.build_model(input_shape)
        
        # Load weights
        model.load_weights(weights_path)
        print(f"Successfully loaded weights from {weights_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise e

def predict_and_save_video(video_path):
    """Main prediction function for Gradio interface"""
    try:
        print(f"Received video path: {video_path}")
        
        # Process input video and get both input and future frames
        input_frames, future_frames = process_video(video_path)
        print(f"Input frames shape: {input_frames.shape}")
        print(f"Future frames shape: {future_frames.shape}")
        
        # Load model and generate predictions
        model = load_model_for_inference("checkpoints/latest_weights.weights.h5")
        predictions = model.predict(input_frames)
        print(f"Generated predictions shape: {predictions.shape}")
        
        # Evaluate predictions
        metrics = evaluate_predictions(input_frames[0], predictions[0], future_frames)
        print("\nPrediction Metrics:")
        print(f"Average SSIM: {metrics['avg_ssim']:.4f}")
        print(f"Average MSE: {metrics['avg_mse']:.4f}")
        print("\nPer-frame SSIM:", [f"{x:.4f}" for x in metrics['ssim_per_frame']])
        print("Per-frame MSE:", [f"{x:.4f}" for x in metrics['mse_per_frame']])
        
        # Generate output videos
        output_path = "predicted_output.mp4"
        comparison_path = "predicted_comparison.mp4"
        video_gen = VideoGenerator()
        
        video_gen.generate_video(input_frames[0], predictions[0], output_path)
        video_gen.generate_merged_video(input_frames[0], predictions[0], comparison_path)
        
        # Create markdown with metrics
        metrics_md = f"""
        ### Prediction Metrics
        - Average SSIM: {metrics['avg_ssim']:.4f}
        - Average MSE: {metrics['avg_mse']:.4f}
        
        #### Per-frame SSIM:
        {' | '.join(f"Frame {i+1}: {x:.4f}" for i, x in enumerate(metrics['ssim_per_frame']))}
        
        #### Per-frame MSE:
        {' | '.join(f"Frame {i+1}: {x:.4f}" for i, x in enumerate(metrics['mse_per_frame']))}
        """
        
        print("Successfully generated prediction videos")
        return [output_path, comparison_path, metrics_md]
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return [None, None, "Error processing video"]

# Create Gradio interface
iface = gr.Interface(
    fn=predict_and_save_video,
    inputs=gr.Video(label="Upload Video"),
    outputs=[
        gr.Video(label="Predicted Future Frames"),
        gr.Video(label="Side-by-Side Comparison"),
        gr.Markdown(label="Prediction Metrics")
    ],
    title="Video Future Frame Prediction with PredRNN",
    description="Upload a video file to predict future frames. The model will process the first 10 frames and predict the next 10 frames.",
    article="""
    ## How it works
    1. Upload a video (at least 20 frames long)
    2. First 10 frames are used as input
    3. Next 10 frames are used as ground truth for evaluation
    4. Model predicts 10 frames
    5. Predictions are evaluated using SSIM and MSE metrics
    
    ## Notes
    - Videos are converted to grayscale and resized to 64x64
    - SSIM ranges from -1 to 1 (higher is better)
    - MSE is always positive (lower is better)
    """
)

if __name__ == "__main__":
    iface.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=True
    )