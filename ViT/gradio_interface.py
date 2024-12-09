import os
import shutil
import numpy as np
import tensorflow as tf
import cv2
import gradio as gr
from predictor import process_and_predict_video
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error
from model_architecture import ViViT
import tempfile
import time

def load_model():
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
    dummy_input = tf.zeros((1, 10, 64, 64, 1))
    _ = model(dummy_input)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='mse')
    model.load_weights("vivit_grayscale_model.keras")
    return model

MODEL = load_model()

def calculate_metrics(predicted_frames, actual_frames):
    ssim_scores = []
    
    # Ensure proper dimensions for both arrays
    pred_frames = np.squeeze(predicted_frames)
    actual_frames = np.squeeze(actual_frames)
    
    for pred_frame, actual_frame in zip(pred_frames, actual_frames):
        ssim_score = ssim(pred_frame, actual_frame, win_size=5, data_range=1.0, channel_axis=None)
        ssim_scores.append(ssim_score)
    
    avg_ssim = np.mean(ssim_scores)
    mse = mean_squared_error(pred_frames.flatten(), actual_frames.flatten())
    
    return avg_ssim, mse

def predict_video(input_video):
    try:
        output_path = "./predicted_output.mp4"
        predictions = process_and_predict_video(
            input_video_path=input_video,
            model=MODEL,
            output_path=output_path
        )
        
        time.sleep(2)
        
        if os.path.exists(output_path):
            first_frame = (predictions[0][0] * 255).astype(np.uint8)
            cv2.imwrite("first_frame.png", first_frame)
            ssim_value, mse_value = calculate_metrics(predictions[0], predictions[1])
            return ["first_frame.png", f"SSIM: {ssim_value:.4f}, MSE: {mse_value:.4f}"]
        else:
            return [None, "Error: Output file not found"]
        
    except Exception as e:
        return [None, f"Error: {str(e)}"]

iface = gr.Interface(
    fn=predict_video,
    inputs=gr.Video(label="Upload Video"),
    outputs=[
        gr.Image(label="Predicted First Frame"),
        gr.Textbox(label="Metrics")
    ],
    title="Video Future Frame Prediction",
    description="Upload a video file to predict future frames.",
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch(server_name="127.0.0.1", server_port=7860, share=True)