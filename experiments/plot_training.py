#!/usr/bin/env python3
"""Plot training progress"""

import torch
import matplotlib.pyplot as plt
import glob
from pathlib import Path
import json

def plot_training_progress(checkpoint_dir):
    """Plot loss over time from checkpoints"""
    
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = sorted(checkpoint_dir.glob('checkpoint_step_*.pt'))
    
    if not checkpoints:
        print("❌ No checkpoints found!")
        return
    
    steps = []
    losses = []
    
    print(f"Found {len(checkpoints)} checkpoints")
    
    for ckpt_path in checkpoints:
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu')
            step = ckpt.get('step', 0)
            loss = ckpt.get('loss', None)
            
            if loss is not None and loss > 0:  # Filter out zero losses
                steps.append(step)
                losses.append(loss)
                print(f"  Step {step}: loss={loss:.4f}")
        except Exception as e:
            print(f"  ⚠️  Could not load {ckpt_path.name}: {e}")
    
    if not steps:
        print("❌ No valid checkpoints with loss data!")
        return
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, marker='o', linewidth=2, markersize=8, color='#e74c3c')
    plt.xlabel('Training Steps', fontsize=12, fontweight='bold')
    plt.ylabel('Loss', fontsize=12, fontweight='bold')
    plt.title('Training Progress', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Save plot
    output_path = checkpoint_dir.parent / 'training_curve.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Plot saved to: {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("📊 Training Summary")
    print("="*60)
    print(f"Total Steps: {max(steps)}")
    print(f"Checkpoints: {len(steps)}")
    
    if len(losses) > 1:
        print(f"Initial Loss: {losses[0]:.4f}")
        print(f"Final Loss: {losses[-1]:.4f}")
        
        if losses[0] > 0:
            reduction = ((losses[0] - losses[-1]) / losses[0] * 100)
            print(f"Loss Reduction: {reduction:.1f}%")
    else:
        print(f"Loss: {losses[0]:.4f}")
    
    print("="*60)
    
    try:
        plt.show()
    except:
        pass  # Skip if no display available

if __name__ == "__main__":
    import sys
    
    checkpoint_dir = sys.argv[1] if len(sys.argv) > 1 else "experiments/quick_test/checkpoints"
    plot_training_progress(checkpoint_dir)
