# config.py
import os
import torch

ROOT_DIR = r"C:\University\ConvLSTM"
DATA_DIR = os.path.join(ROOT_DIR, "train")
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Class Configuration
SELECTED_CLASSES = ['WalkingWithDog', 'Basketball', 'Bowling', 'JumpingJack', 'Biking']

# Model Configuration
INPUT_CHANNELS = 1
HIDDEN_CHANNELS = 64
KERNEL_SIZE = (3, 3)
INPUT_TIMESTEPS = 10
OUTPUT_TIMESTEPS = 10
IMG_SIZE = (64, 64)

# Training Configuration
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')