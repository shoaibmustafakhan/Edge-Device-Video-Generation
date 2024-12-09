# train.py
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from config import *
from utils import calculate_accuracy, save_checkpoint

def train_model(model, train_loader, val_loader, criterion, optimizer, start_epoch=0, num_epochs=NUM_EPOCHS):
    print(f"Training on device: {DEVICE}")
    scaler = GradScaler('cuda')
    
    for epoch in range(start_epoch, num_epochs):
        print(f"\nStarting Epoch {epoch + 1}/{num_epochs}...")
        model.train()
        train_loss = 0
        
        for batch_idx, (X_batch, Y_batch) in enumerate(train_loader):
            X_batch, Y_batch = X_batch.to(DEVICE), Y_batch.to(DEVICE)

            optimizer.zero_grad()
            with autocast('cuda'):
                outputs = model(X_batch)
                loss = criterion(outputs, Y_batch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            if (batch_idx + 1) % 10 == 0:
                print(f"Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        train_loss /= len(train_loader)
        print(f"Epoch {epoch + 1} Train Loss: {train_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        val_accuracy = 0
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch, Y_batch = X_batch.to(DEVICE), Y_batch.to(DEVICE)
                with autocast('cuda'):
                    outputs = model(X_batch)
                    loss = criterion(outputs, Y_batch)
                val_loss += loss.item()
                val_accuracy += calculate_accuracy(outputs, Y_batch)

        val_loss /= len(val_loader)
        val_accuracy /= len(val_loader)
        print(f"Epoch {epoch + 1} Validation Loss: {val_loss:.4f}, Accuracy (SSIM): {val_accuracy:.4f}")

        # Save checkpoint after every epoch
        save_checkpoint(model, optimizer, epoch + 1, val_loss)