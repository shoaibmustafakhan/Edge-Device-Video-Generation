import cv2
import numpy as np
import os

class VideoGenerator:
    @staticmethod
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
        
        # Initialize video writer
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

    @staticmethod
    def generate_video(input_frames, predicted_frames, output_filename, duration=3, fps=20):
        # Combine input and predicted frames
        all_frames = np.concatenate([input_frames, predicted_frames], axis=0)
        
        # Save combined frames as video
        VideoGenerator.save_frames_as_video(all_frames, output_filename, fps)

    @staticmethod
    def generate_merged_video(input_frames, predicted_frames, output_filename, duration=5, fps=20):
        """Generate side-by-side comparison video"""
        # Ensure proper shape
        input_frames = np.squeeze(input_frames)
        predicted_frames = np.squeeze(predicted_frames)
        
        # Create side-by-side frames
        combined_frames = []
        max_frames = max(len(input_frames), len(predicted_frames))
        
        for i in range(max_frames):
            # Get frames (loop if needed)
            input_frame = input_frames[i % len(input_frames)]
            pred_frame = predicted_frames[i % len(predicted_frames)]
            
            # Convert to uint8 if needed
            if input_frame.dtype != np.uint8:
                input_frame = (input_frame * 255).astype(np.uint8)
            if pred_frame.dtype != np.uint8:
                pred_frame = (pred_frame * 255).astype(np.uint8)
            
            # Stack horizontally
            combined = np.hstack([input_frame, pred_frame])
            combined_frames.append(combined)
        
        # Convert to numpy array
        combined_frames = np.array(combined_frames)
        
        # Save as video
        VideoGenerator.save_frames_as_video(combined_frames, output_filename, fps)