import os
import numpy as np
import tensorflow as tf
from data_loader import prepare_dataset, VideoFrameDataset
from model_architecture import ViViT
import json
from training_evaluation import train_model, plot_training_history, predict_future_frames, visualize_predictions, save_frames_as_video

def main():
    # First, verify GPU is available
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("TensorFlow version:", tf.__version__)

    # Prepare datasets
    print("\nPreparing datasets...")
    train_path, val_path, test_path = prepare_dataset()

    # Create datasets
    print("\nCreating data loaders...")
    train_dataset = VideoFrameDataset(train_path, input_frames=10, future_frames=10, batch_size=4)
    val_dataset = VideoFrameDataset(val_path, input_frames=10, future_frames=10, batch_size=4)

    # Create model
    print("\nInitializing ViViT model...")
    model = ViViT(
        input_shape=(10, 64, 64, 1),  # 10 input frames, 64x64 resolution, 1 channel (grayscale)
        patch_size=8,
        embed_dim=256,
        num_heads=16,
        ff_dim=512,
        num_transformer_layers=8,
        dropout=0.1,
        future_frames=10
    )

    # Train model
    print("\nStarting training...")
    history = train_model(model, train_dataset, val_dataset, epochs=60)

    # Plot training history
    print("\nPlotting training history...")
    plot_training_history(history)

    # Test prediction on a sample video
    print("\nTesting prediction on sample video...")
    test_video_path = os.path.join(test_path, "Bowling/v_Bowling_g01_c01.npy")
    if os.path.exists(test_video_path):
        test_video = np.load(test_video_path)
        test_video = np.mean(test_video, axis=-1, keepdims=True)
        input_frames = test_video[:10]
        
        predicted_frames = predict_future_frames(model, input_frames)
        
        # Visualize results
        visualize_predictions(input_frames, predicted_frames[0])
        
        # Save as video
        save_frames_as_video(predicted_frames[0], "predicted_video.mp4")
        print("\nPrediction video saved as 'predicted_video.mp4'")
    else:
        print(f"\nTest video not found at {test_video_path}")

    print("\nDone!")

if __name__ == "__main__":
    main()