# preprocessor.py
import numpy as np
import logging

class Preprocessor:
    def __init__(self, input_sequence_length=10, output_sequence_length=10):
        self.input_sequence_length = input_sequence_length
        self.output_sequence_length = output_sequence_length
        self.logger = logging.getLogger(__name__)
        
    def prepare_data(self, batch_videos):
        """
        Prepare data from batch of videos, handling variable length videos.
        Args:
            batch_videos: List of video frames arrays
        Returns:
            X, Y: Training sequences
        """
        X, Y = [], []
        
        for video_frames in batch_videos:
            # Convert to numpy array if not already
            video_frames = np.array(video_frames)
            
            # Skip if video is too short for sequence creation
            required_length = self.input_sequence_length + self.output_sequence_length
            if len(video_frames) < required_length:
                self.logger.warning(f"Skipping video with insufficient frames: {len(video_frames)} < {required_length}")
                continue
                
            # Create sequences from this video
            for i in range(len(video_frames) - required_length + 1):
                input_seq = video_frames[i:i + self.input_sequence_length]
                output_seq = video_frames[i + self.input_sequence_length:
                                        i + self.input_sequence_length + self.output_sequence_length]
                
                X.append(input_seq)
                Y.append(output_seq)
        
        if not X or not Y:
            raise ValueError("No valid sequences could be created from the batch")
            
        # Convert to numpy arrays and add channel dimension
        X = np.array(X)[..., np.newaxis]  # Shape: (samples, input_length, 64, 64, 1)
        Y = np.array(Y)[..., np.newaxis]  # Shape: (samples, output_length, 64, 64, 1)
        
        self.logger.info(f"Created sequences - X shape: {X.shape}, Y shape: {Y.shape}")
        return X, Y