# main.py
import torch.nn as nn
import torch.optim as optim
from config import *
from dataset import get_dataloaders
from model import ConvLSTM
from train import train_model
from utils import load_checkpoint

def main():
    print(f"Using device: {DEVICE}")
    
    # Get dataloaders
    train_loader, val_loader = get_dataloaders(DATA_DIR)
    
    # Initialize model, criterion, and optimizer
    model = ConvLSTM().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Load checkpoint if exists
    start_epoch = load_checkpoint(model, optimizer)
    
    # Train model
    train_model(model, train_loader, val_loader, criterion, optimizer, start_epoch=start_epoch)

if __name__ == "__main__":
    main()