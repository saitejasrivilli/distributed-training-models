import torch
import os
from pathlib import Path

def save_checkpoint(model, optimizer, step, loss, output_dir, rank=0):
    """Save training checkpoint"""
    if rank != 0:
        return
    
    checkpoint_dir = Path(output_dir) / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Get model state dict (unwrap DDP if needed)
    if hasattr(model, 'module'):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    
    checkpoint = {
        'step': step,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': float(loss),  # Ensure it's a float
    }
    
    checkpoint_path = checkpoint_dir / f'checkpoint_step_{step}.pt'
    torch.save(checkpoint, checkpoint_path)
    
    print(f"✅ Checkpoint saved: {checkpoint_path} (loss: {loss:.4f})")
    
    return checkpoint_path

def load_checkpoint(model, optimizer, checkpoint_path):
    """Load training checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['step'], checkpoint['loss']
