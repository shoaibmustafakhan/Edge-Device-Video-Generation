import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
from model_architecture import PatchEmbedding, TransformerBlock, ViViT
import os

def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

def combined_loss(y_true, y_pred):
    mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    ssim = ssim_loss(y_true, y_pred)
    return 0.7 * mse + 0.3 * ssim

def train_model(model, train_dataset, val_dataset, epochs=60):
    # Build the model first
    dummy_input = tf.zeros((1, 10, 64, 64, 1))
    _ = model(dummy_input)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss=combined_loss, metrics=["mse", ssim_loss])
    
    checkpoint_path = "vivit_grayscale_model.keras"
    history_path = "training_history.json"
    current_epoch = 0
    
    # Load existing weights and history if they exist
    if os.path.exists(checkpoint_path):
        print(f"\nFound existing weights at {checkpoint_path}")
        print("Loading weights and continuing training...")
        model.load_weights(checkpoint_path)
        
        # Load training history if it exists
        if os.path.exists(history_path):
            try:
                with open(history_path, 'r') as f:
                    history_dict = json.loads(f.read())
                    if 'loss' in history_dict:
                        current_epoch = len(history_dict['loss'])
                        print(f"Resuming from epoch {current_epoch}")
            except Exception as e:
                print(f"Could not load history file: {e}")
                print("Starting from epoch 0")
                current_epoch = 0
    
    class SaveHistoryCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if hasattr(self.model, 'history') and self.model.history is not None:
                # Save history after each epoch
                history_dict = self.model.history.history
                with open(history_path, 'w') as f:
                    json.dump(history_dict, f)
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        initial_epoch=current_epoch,  # Start from the last completed epoch
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss'),
            SaveHistoryCallback()
        ]
    )
    return history

def plot_training_history(history):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss (MSE + SSIM)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mse'], label='Train MSE')
    plt.plot(history.history['val_mse'], label='Val MSE')
    plt.title('Mean Squared Error (MSE)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()

    plt.tight_layout()
    plt.show()

def predict_future_frames(model, input_sequence):
    input_sequence = np.expand_dims(input_sequence, axis=0)
    input_sequence = input_sequence.astype(np.float32) / 255.0
    
    predicted_frames = model.predict(input_sequence)
    predicted_frames = np.clip(predicted_frames, 0, 1)
    predicted_frames = (predicted_frames * 255).astype(np.uint8)
    
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

def save_frames_as_video(frames, output_path, fps=1):
    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height), isColor=False)
    
    for frame in frames:
        frame = np.squeeze(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        out.write(frame)
    
    out.release()