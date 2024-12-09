import pandas as pd
import os

#loading into datasets

# Define paths
dataset_path = "/kaggle/input/ucf101-action-recognition"

train_csv = os.path.join(dataset_path, "train.csv")
test_csv=os.path.join(dataset_path,"test.csv")
val_csv=os.path.join(dataset_path,"val.csv")

#--------------------------------------------
#load into dataframes
train_df = pd.read_csv(train_csv)
test_df=pd.read_csv(test_csv)
val_df=pd.read_csv(val_csv)

# for now using only one class -> jumping jacks
jumping_jacks_df = train_df[train_df['label'] == 'PullUps']

# Get the total number of videos for "Jumping Jacks"
total_videos = len(jumping_jacks_df)
print(f"Total videos for 'JumpingJack': {total_videos}")

# Store paths of all 92 videos
video_paths = jumping_jacks_df['clip_path'].tolist()

# Output the full paths
for idx, video_path in enumerate(video_paths, 1):
    full_path = os.path.join(dataset_path, video_path)
    print(f"Video {idx}: {full_path}")


import cv2
import os
import matplotlib.pyplot as plt

# Define dataset path and video folder
dataset_path = "/kaggle/input/ucf101-action-recognition"
jumping_jacks_folder = "train/PullUps"  # Folder containing the JumpingJack videos

# Initialize a list to store all frames from all videos
jumpingjack_frames = []

# Loop over all videos in the JumpingJack class
for video_name in os.listdir(os.path.join(dataset_path, jumping_jacks_folder)):
    video_path = os.path.join(jumping_jacks_folder, video_name)
    full_video_path = os.path.join(dataset_path, video_path)

    if os.path.exists(full_video_path):
        print(f"Processing video: {video_name}")
    else:
        print(f"Video does not exist: {video_name}")
        continue

    # Try to open the video using OpenCV
    cap = cv2.VideoCapture(full_video_path)

    if cap.isOpened():
        print(f"Video {video_name} opened successfully!")

        # Initialize a list to store frames for the current video
        frames = []

        # Read frames one by one
        while True:
            ret, frame = cap.read()

            if not ret:
                break  # Break if we reach the end of the video

            # Resize frame to 64x64
            frame_resized = cv2.resize(frame, (64, 64))

            # Convert frame to grayscale
            frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

            # Append the processed frame to the current list
            frames.append(frame_gray)

        # Append frames of this video to the all_frames list
        jumpingjack_frames.append(frames)

        print(f"Total frames processed for {video_name}: {len(frames)}")

    else:
        print(f"Failed to open video {video_name}")

# Now, all_frames contains the frames of all the videos
print(f"Total videos processed: {len(jumpingjack_frames)}")

# Optional: Visualize a few frames from the first video
if jumpingjack_frames:
    fig, axes = plt.subplots(1, 5, figsize=(15, 5))
    for i, ax in enumerate(axes):
        ax.imshow(jumpingjack_frames[0][i], cmap='gray')
        ax.axis('off')
    plt.show()




import numpy as np

# Number of input frames (for short sequence)
input_sequence_length = 10

# Number of frames to predict (future frames)
output_sequence_length = 5

# Prepare input-output pairs for training
def create_sequences(frames, input_length, output_length):
    input_sequences = []
    output_sequences = []

    # Ensure that we have enough frames for the sequence
    for i in range(len(frames) - input_length - output_length):
        input_seq = frames[i:i + input_length]
        output_seq = frames[i + input_length:i + input_length + output_length]

        input_sequences.append(input_seq)
        output_sequences.append(output_seq)

    return np.array(input_sequences), np.array(output_sequences)

# Example for one video (jumping jack frames)
input_frames = jumpingjack_frames[0]  # Choose the first video
X, Y = create_sequences(input_frames, input_sequence_length, output_sequence_length)

print(f"Input shape: {X.shape}, Output shape: {Y.shape}")



from keras.layers import Conv3D, ConvLSTM2D, Input, TimeDistributed, Flatten, Dense
from keras.models import Model

def build_predrnn(input_shape):
    """
    Builds a simple PredRNN-like model for video frame prediction.

    :param input_shape: Tuple, shape of the input (sequence_length, height, width, channels)
    :return: Keras Model object
    """
    # Define input
    inputs = Input(shape=input_shape)

    # First ConvLSTM2D layer (spatial-temporal modeling)
    x = ConvLSTM2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
        activation="relu"
    )(inputs)

    # 3D Convolutional layer for temporal prediction
    x = Conv3D(
        filters=64,
        kernel_size=(3, 3, 3),  # (temporal depth, height, width)
        padding="same",
        activation="relu"
    )(x)

    # Output layer
    outputs = Conv3D(
        filters=1,  # Output channels (e.g., grayscale)
        kernel_size=(3, 3, 3),
        padding="same",
        activation="sigmoid"
    )(x)

    # Create the model
    model = Model(inputs, outputs)
    return model


# Build the PredRNN model
input_shape = (input_sequence_length, 64, 64, 1)  # 10 frames of 64x64 with 1 channel (grayscale)
model = build_predrnn(input_shape)

# Summary of the model
model.summary()




import numpy as np

# Example: Splitting video frames into input-output sequences
def prepare_data(frames, input_len=10, output_len=5):
    """
    Splits video frames into input and output sequences for training.

    :param frames: List of all frames (grayscale 64x64).
    :param input_len: Number of frames in the input sequence.
    :param output_len: Number of frames in the output sequence.
    :return: Tuple of (X, Y) for training.
    """
    X, Y = [], []
    total_frames = len(frames)

    for i in range(total_frames - input_len - output_len + 1):
        # Input: 10 frames
        X.append(frames[i : i + input_len])

        # Output: Next 5 frames
        Y.append(frames[i + input_len : i + input_len + output_len])

    # Convert to numpy arrays and add channel dimension (grayscale)
    X = np.expand_dims(np.array(X), axis=-1)  # Shape: (samples, input_len, 64, 64, 1)
    Y = np.expand_dims(np.array(Y), axis=-1)  # Shape: (samples, output_len, 64, 64, 1)
    return X, Y

# Example: Prepare training data from `jumpingjack_frames`
all_video_frames = [frame for video in jumpingjack_frames for frame in video]  # Flatten all videos
input_len = 10
# Update prepare_data to set output_len=10
output_len = 10  # Change target sequence length

# Regenerate X and Y with new output_len
X, Y = prepare_data(all_video_frames, input_len=10, output_len=10)

print(f"X shape: {X.shape}, Y shape: {Y.shape}")





# Compile the model
model.compile(
    optimizer="adam",
    loss="mse",  # Mean Squared Error for regression-like predictions
    metrics=["mae"]  # Mean Absolute Error
)

# Train the model
history = model.fit(
    X, Y,
    epochs=20,  # Adjust epochs as needed
    batch_size=8,  # Adjust batch size based on GPU memory
    validation_split=0.2  # Use 20% of data for validation
)
model.save("predrnn_model.h5")





from sklearn.metrics import mean_squared_error
import tensorflow as tf
import numpy as np
import cv2

# Prepare test data (use frames from your dataset not used in training)
test_sequence = jumpingjack_frames[:5]  # Example: Use 5 sequences for testing
test_input = []
test_target = []

input_sequence_length = 10
predicted_sequence_length = 5

for sequence in test_sequence:
    for i in range(len(sequence) - input_sequence_length - predicted_sequence_length):
        input_frames = sequence[i:i+input_sequence_length]
        target_frames = sequence[i+input_sequence_length:i+input_sequence_length+predicted_sequence_length]
        
        test_input.append(np.array(input_frames))
        test_target.append(np.array(target_frames))

test_input = np.array(test_input).astype("float32") / 255.0  # Normalize
test_input = np.expand_dims(test_input, -1)  # Add channel dimension

test_target = np.array(test_target).astype("float32") / 255.0
test_target = np.expand_dims(test_target, -1)  # Add channel dimension

# Predict using the trained model
predicted_frames = model.predict(test_input)

# Evaluate the predictions
mse_scores = []
ssim_scores = []

for i in range(len(test_target)):
    for j in range(predicted_sequence_length):
        target_frame = test_target[i, j]
        predicted_frame = predicted_frames[i, j]
        
        # Calculate MSE
        mse = mean_squared_error(target_frame.flatten(), predicted_frame.flatten())
        mse_scores.append(mse)
        
        # Calculate SSIM
        ssim = tf.image.ssim(target_frame, predicted_frame, max_val=1.0).numpy()
        ssim_scores.append(ssim)

# Average metrics
avg_mse = np.mean(mse_scores)
avg_ssim = np.mean(ssim_scores)

print(f"Average MSE: {avg_mse}")
print(f"Average SSIM: {avg_ssim}")




import numpy as np
import cv2

def generate_video(input_frames, predicted_frames, output_filename, duration=3, fps=20):
    """
    Generate a video of a specific duration using input and predicted frames.
    
    Args:
    - input_frames: Array of input frames (e.g., X_test[0]), normalized to [0, 1].
    - predicted_frames: Array of predicted frames (e.g., predictions[0]), normalized to [0, 1].
    - output_filename: Name of the output video file.
    - duration: Desired duration of the video in seconds (default 3).
    - fps: Frames per second for the video (default 20).
    """
    # Total frames required
    total_frames = duration * fps
    
    # Concatenate input and predicted frames
    frames = np.concatenate((input_frames, predicted_frames), axis=0)
    
    # Ensure the total frames are sufficient
    if len(frames) < total_frames:
        # Tile the frames to fill up the total duration
        frames = np.tile(frames, (total_frames // len(frames) + 1, 1, 1, 1))
    frames = frames[:total_frames]  # Trim to exact number of frames
    
    # Convert frames from [0, 1] to [0, 255] and ensure type is uint8
    frames = np.clip(frames, 0, 1)  # Ensure values are within [0, 1]
    frames = (frames * 255).astype(np.uint8)
    
    # Initialize VideoWriter
    if len(frames[0].shape) == 2:  # Grayscale frames
        frame_height, frame_width = frames[0].shape
        isColor = False
    elif len(frames[0].shape) == 3:  # Color frames
        frame_height, frame_width, _ = frames[0].shape
        isColor = True
    else:
        raise ValueError("Unexpected frame dimensions. Frames must be 2D (grayscale) or 3D (color).")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height), isColor=isColor)
    
    # Process and write each frame to the video
    for frame in frames:
        if len(frame.shape) == 2:  # If the frame is grayscale, convert it to BGR
            frame_colored = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            frame_colored = frame  # Already in color
        
        # Write the frame to the video
        video_writer.write(frame_colored)
    
    # Release the video writer
    video_writer.release()
    print(f"Video saved as {output_filename}")



#generate_video(X_test[0], predictions[0], 'test_video_output_3s.mp4', duration=5, fps=20)





def generate_merged_video(input_frames, predicted_frames, output_filename, duration=5, fps=20):
    """
    Generate a video showing input and predicted frames side-by-side.

    Args:
    - input_frames: Array of input frames (e.g., X_test[0]).
    - predicted_frames: Array of predicted frames (e.g., predictions[0]).
    - output_filename: Name of the output video file.
    - duration: Desired duration of the video in seconds (default 5).
    - fps: Frames per second for the video (default 20).
    """
    total_frames = duration * fps
    frames = np.concatenate((input_frames, predicted_frames), axis=0)
    
    # Repeat frames to reach the required total
    while len(frames) < total_frames:
        frames = np.concatenate((frames, frames), axis=0)
    frames = frames[:total_frames]  # Trim to exact number of frames

    # Convert frames to uint8 (0-255 range)
    frames = (frames * 255).astype(np.uint8)

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (128, 64))  # Width is doubled for side-by-side

    for i in range(total_frames):
        input_frame = input_frames[i % len(input_frames)]  # Loop through input frames
        predicted_frame = predicted_frames[i % len(predicted_frames)]  # Loop through predicted frames

        # Resize frames to ensure same dimensions
        input_frame_resized = cv2.resize(input_frame, (64, 64))
        predicted_frame_resized = cv2.resize(predicted_frame, (64, 64))

        # Convert frames to uint8 if not already
        input_frame_resized = (input_frame_resized * 255).astype(np.uint8)
        predicted_frame_resized = (predicted_frame_resized * 255).astype(np.uint8)

        # Convert grayscale to BGR
        input_colored = cv2.cvtColor(input_frame_resized, cv2.COLOR_GRAY2BGR)
        predicted_colored = cv2.cvtColor(predicted_frame_resized, cv2.COLOR_GRAY2BGR)

        # Combine input and predicted frames side-by-side
        combined_frame = np.hstack((input_colored, predicted_colored))
        video_writer.write(combined_frame)

    video_writer.release()
    print(f"5-second merged video saved as {output_filename}")

#generate_merged_video(X_test[1], predictions[1], 'test_video2_merged_output.mp4', duration=5, fps=20)






import cv2
import numpy as np
from keras.models import load_model
from keras.metrics import MeanSquaredError  # Import the metric used in the model

# Define custom_objects to include any custom metrics or losses
custom_objects = {
    'mse': MeanSquaredError()  # Register the metric
}

# Load the model with custom_objects
model_path = '/kaggle/input/pullup_predrnn/keras/default/1/predrnn_model_pullup.h5'
model = load_model(model_path, custom_objects=custom_objects)
# Function to load video and extract frames
def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized_frame = cv2.resize(gray_frame, (64, 64))  # Ensure it matches model input size
        frames.append(resized_frame)

    cap.release()
    return frames

# Function to prepare data for predictions
def prepare_data(frames, input_len=10, output_len=5):
    X, Y = [], []
    total_frames = len(frames)

    for i in range(total_frames - input_len - output_len + 1):
        X.append(frames[i : i + input_len])
        Y.append(frames[i + input_len : i + input_len + output_len])

    X = np.expand_dims(np.array(X), axis=-1)
    Y = np.expand_dims(np.array(Y), axis=-1)
    return X, Y

# Load test video
test_video_path = '/kaggle/input/ucf101-action-recognition/test/PullUps/v_PullUps_g07_c02.avi'  # Update path
test_video_frames = load_video(test_video_path)

# Prepare the test data
X_test, Y_test = prepare_data(test_video_frames, input_len=10, output_len=5)

print(f"X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}")

# Generate predictions using the loaded model
predictions = model.predict(X_test)

print(f"Predictions shape: {predictions.shape}")




import cv2
import numpy as np

def load_video_to_frames(video_path, timesteps=10):
    """
    Load a video, preprocess its frames, and prepare it for model input.

    Args:
    - video_path: Path to the test video.
    - timesteps: Number of frames in each sequence (default is 10).

    Returns:
    - X_test: Numpy array of shape (1, timesteps, 64, 64, 1) ready for prediction.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize to 64x64 and convert to grayscale
        frame = cv2.resize(frame, (64, 64))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Normalize pixel values to [0, 1]
        frame = frame / 255.0
        frames.append(frame)

    cap.release()

    # Ensure we have enough frames for timesteps
    if len(frames) < timesteps:
        raise ValueError(f"Video has fewer than {timesteps} frames!")
    
    # Create sequences of length `timesteps`
    frames = np.array(frames)
    frames = frames[:timesteps]  # Use the first `timesteps` frames
    frames = np.expand_dims(frames, axis=-1)  # Add channel dimension

    # Add batch dimension to make shape (1, timesteps, 64, 64, 1)
    X_test = np.expand_dims(frames, axis=0)

    return X_test

# Example usage:
test_video_path = '/kaggle/input/ucf101-action-recognition/test/PullUps/v_PullUps_g06_c04.avi'  # Update with your test video path
X_test = load_video_to_frames(test_video_path, timesteps=10)

# Generate predictions using the model
predictions = model.predict(X_test)

# Create the video combining input and predicted frames
generate_video(X_test[0], predictions[0], 'test_video_output_3s.mp4', duration=5, fps=20)
#generate_merged_video(X_test[0], predictions[0], 'test_video2_merged_output.mp4', duration=5, fps=20)




