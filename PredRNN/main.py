import os
import logging
from datetime import datetime
import numpy as np
from dataloader import DataLoader
from preprocessor import Preprocessor
from model import PredRNN
from video_generator import VideoGenerator
import tensorflow as tf
import gc
import time
import psutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_ram_usage():
    """Get current RAM usage in GB"""
    process = psutil.Process(os.getpid())
    ram_usage = process.memory_info().rss / (1024 * 1024 * 1024)  # Convert to GB
    return ram_usage

def check_and_refresh_memory(model_builder, input_shape, latest_weights):
    """Check RAM usage and refresh if over threshold"""
    ram_usage = get_ram_usage()
    if ram_usage > 22:  # 22 GB threshold
        logger.info(f"RAM usage ({ram_usage:.2f} GB) exceeded 22 GB threshold. Refreshing system...")
        
        # Save any necessary state
        if not latest_weights:
            latest_weights = os.path.join('checkpoints', 'latest_weights.weights.h5')
        
        # Clear everything
        clear_memory()
        
        # Wait a moment for memory to clear
        time.sleep(2)
        
        # Reload model and weights
        logger.info("Reloading model and weights...")
        model = load_model_state(model_builder, input_shape, latest_weights)
        
        logger.info(f"System refreshed. Current RAM usage: {get_ram_usage():.2f} GB")
        return model
    return None

def setup_directories():
    dirs = ['checkpoints', 'outputs', 'logs']
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)
            logger.info(f"Created directory: {dir}")

def clear_memory():
    """Clear memory and GPU memory if available"""
    # Clear Keras backend session
    tf.keras.backend.clear_session()
    
    # Force garbage collection
    gc.collect()
    
    # Clear GPU memory if available
    if tf.config.list_physical_devices('GPU'):
        try:
            for device in tf.config.list_physical_devices('GPU'):
                tf.config.experimental.reset_memory_stats(device)
        except:
            pass

def find_latest_weights():
    """Find the latest weights file in checkpoints directory"""
    weights_path = os.path.join('checkpoints', 'latest_weights.weights.h5')
    if os.path.exists(weights_path):
        return weights_path
    return None

def load_model_state(model_builder, input_shape, latest_weights=None):
    """Load model with optional weights"""
    model = model_builder.build_model(input_shape)
    if latest_weights and os.path.exists(latest_weights):
        try:
            model.load_weights(latest_weights)
            logger.info(f"Loaded weights from: {latest_weights}")
        except Exception as e:
            logger.warning(f"Could not load weights: {str(e)}")
    return model

def save_model_state(model, epoch, video_index, timestamp):
    """Save model weights"""
    weights_path = os.path.join('checkpoints', 'latest_weights.weights.h5')
    model.save_weights(weights_path)
    return weights_path

def main():
    setup_directories()
    
    # Initialize components
    logger.info("Initializing components...")
    dataloader = DataLoader()
    preprocessor = Preprocessor()
    model_builder = PredRNN()
    video_generator = VideoGenerator()
    
    # Initial model build to get architecture
    input_shape = (10, 64, 64, 1)
    
    # Check for existing weights
    latest_weights = find_latest_weights()
    if latest_weights:
        logger.info(f"Found existing weights at: {latest_weights}")
        logger.info("Will continue training from last checkpoint")
    else:
        logger.info("No existing weights found. Starting fresh training.")
    
    # Training loop
    epochs = 20
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1}/{epochs}")
        video_index = 0
        
        while True:
            # Monitor and refresh memory if needed
            refreshed_model = check_and_refresh_memory(model_builder, input_shape, latest_weights)
            
            # Get next video
            frames, current_class = dataloader.next_video()
            
            if frames is None:
                logger.info("Completed processing all videos in this epoch")
                break
            
            try:
                # Load fresh model instance or use refreshed one
                if refreshed_model is None:
                    logger.info("Loading fresh model instance...")
                    model = load_model_state(model_builder, input_shape, latest_weights)
                else:
                    model = refreshed_model
                
                # Create sequences from single video
                X, Y = preprocessor.prepare_data([frames])
                
                # Train on single video
                logger.info(f"Training on video {video_index + 1} for class {current_class}")
                history = model.train_on_batch(X, Y)
                logger.info(f"Loss: {history[0]:.4f}")
                logger.info(f"Current RAM usage: {get_ram_usage():.2f} GB")
                
                # Save current model state
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                latest_weights = save_model_state(model, epoch + 1, video_index + 1, timestamp)
                logger.info(f"Saved weights to: {latest_weights}")
                
                # Clear everything from memory
                logger.info("Clearing memory...")
                del X, Y, frames, model
                clear_memory()
                
                # Wait for a second
                logger.info("Waiting for 1 second before next video...")
                time.sleep(1)
                
                video_index += 1
                
            except Exception as e:
                logger.error(f"Error processing video: {str(e)}")
                continue
        
        # Save final model after each epoch
        logger.info(f"Saving final model for epoch {epoch+1}")
        final_model = load_model_state(model_builder, input_shape, latest_weights)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join('checkpoints', f'model_epoch_{epoch+1}_{timestamp}.keras')
        final_model.save(model_path)
        logger.info(f"Saved complete model for epoch {epoch+1} to: {model_path}")
        
        # Clear final model from memory
        del final_model
        clear_memory()
    
    logger.info("Training pipeline completed successfully!")

if __name__ == "__main__":
    main()