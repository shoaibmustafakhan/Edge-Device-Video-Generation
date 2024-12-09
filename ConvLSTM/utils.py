# utils.py
from skimage.metrics import structural_similarity as ssim
import torch
import os
from config import CHECKPOINT_DIR

def calculate_accuracy(y_pred, y_true):
    """Calculate accuracy using Structural Similarity Index (SSIM)."""
    y_pred = y_pred.cpu().numpy()
    y_true = y_true.cpu().numpy()
    batch_ssim = 0
    for b in range(y_pred.shape[0]):
        for t in range(y_pred.shape[2]):
            batch_ssim += ssim(y_true[b, 0, t], y_pred[b, 0, t], data_range=1.0)
    return batch_ssim / (y_pred.shape[0] * y_pred.shape[2])

def save_checkpoint(model, optimizer, epoch, loss):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "checkpoint.pth")
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved. Epoch: {epoch}, Loss: {loss:.4f}")

def load_checkpoint(model, optimizer):
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "checkpoint.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming from epoch {start_epoch}")
        return start_epoch
    else:
        print("No checkpoint found. Starting from scratch.")
        return 0