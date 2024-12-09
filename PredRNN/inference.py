import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error
import argparse
import os
from model import PredRNN
from video_generator import VideoGenerator  # Add this import

class Inference:
    def __init__(self, weights_path):
        # Initialize model architecture
        model_builder = PredRNN()
        input_shape = (10, 64, 64, 1)
        self.model = model_builder.build_model(input_shape)
        
        # Load weights
        try:
            self.model.load_weights(weights_path)
            print(f"Successfully loaded weights from: {weights_path}")
        except Exception as e:
            raise Exception(f"Failed to load weights from {weights_path}: {str(e)}")
        
    def load_video_to_frames(self, video_path, timesteps=10):
        """Load a video file and prepare it for prediction."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.resize(frame, (64, 64))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = frame / 255.0
            frames.append(frame)
            
        cap.release()
        
        if len(frames) < timesteps:
            raise ValueError(f"Video has fewer than {timesteps} frames!")
        
        frames = np.array(frames[:timesteps])
        frames = np.expand_dims(frames, axis=-1)
        return np.expand_dims(frames, axis=0)
    
    def predict_sequence(self, input_sequence):
        return self.model.predict(input_sequence)

def main():
    parser = argparse.ArgumentParser(description='Run video prediction inference')
    parser.add_argument('--weights_path', type=str, required=True, help='Path to the saved model weights')
    parser.add_argument('--video_path', type=str, required=True, help='Path to the input video')
    parser.add_argument('--output_path', type=str, default='prediction_output.mp4', help='Path for the output video')
    args = parser.parse_args()
    
    # Initialize inference
    inference = Inference(args.weights_path)
    
    # Load and prepare video
    input_sequence = inference.load_video_to_frames(args.video_path)
    
    # Generate predictions
    predictions = inference.predict_sequence(input_sequence)
    
    # Generate output video using simplified approach
    video_gen = VideoGenerator()
    video_gen.generate_video(input_sequence[0], predictions[0], args.output_path)
    
    # Generate side-by-side comparison if desired
    comparison_path = args.output_path.replace('.mp4', '_comparison.mp4')
    video_gen.generate_merged_video(input_sequence[0], predictions[0], comparison_path)
    
    print(f"Generated prediction videos at: {args.output_path} and {comparison_path}")

if __name__ == "__main__":
    main()