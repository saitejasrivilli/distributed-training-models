#!/usr/bin/env python3
"""Create system architecture diagram"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(14, 10), facecolor='white')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(5, 9.5, 'Distributed Training System Architecture', 
        ha='center', fontsize=24, fontweight='bold')

# Main Coordinator
coordinator = patches.FancyBboxPatch((2, 7.5), 6, 1.2,
                                      boxstyle="round,pad=0.05",
                                      edgecolor='#3498db', 
                                      facecolor='#ecf0f1',
                                      linewidth=3)
ax.add_patch(coordinator)
ax.text(5, 8.35, 'Training Coordinator', ha='center', fontsize=16, fontweight='bold')
ax.text(5, 7.9, 'Gradient Aggregation • Checkpointing • Monitoring', 
        ha='center', fontsize=11, style='italic')

# GPU Workers
gpu_y = 5.0
gpu_width = 2.0
gpu_height = 1.5
colors = ['#e74c3c', '#f39c12', '#2ecc71', '#9b59b6']
gpu_names = ['GPU 0', 'GPU 1', 'GPU 2', 'GPU 3']

for i in range(4):
    x = 0.5 + i * 2.3
    
    gpu = patches.FancyBboxPatch((x, gpu_y), gpu_width, gpu_height,
                                  boxstyle="round,pad=0.05",
                                  edgecolor=colors[i], 
                                  facecolor='white',
                                  linewidth=2.5)
    ax.add_patch(gpu)
    
    ax.text(x + gpu_width/2, gpu_y + 1.2, gpu_names[i], 
            ha='center', fontsize=14, fontweight='bold', color=colors[i])
    ax.text(x + gpu_width/2, gpu_y + 0.85, 'Model Copy', 
            ha='center', fontsize=11)
    ax.text(x + gpu_width/2, gpu_y + 0.55, 'Local Batch', 
            ha='center', fontsize=10, style='italic')
    ax.text(x + gpu_width/2, gpu_y + 0.25, '(32 batches)', 
            ha='center', fontsize=9, color='gray')
    
    # Arrows from coordinator
    ax.arrow(5, 7.5, (x + gpu_width/2 - 5) * 0.8, -0.4,
             head_width=0.15, head_length=0.1, fc='#95a5a6', ec='#95a5a6', 
             linewidth=2, alpha=0.6)

# NCCL Communication
nccl_box = patches.FancyBboxPatch((1, 4.0), 8, 0.6,
                                   boxstyle="round,pad=0.03",
                                   edgecolor='#16a085', 
                                   facecolor='#d5f4e6',
                                   linewidth=2)
ax.add_patch(nccl_box)
ax.text(5, 4.3, 'NCCL All-Reduce (Gradient Synchronization)', 
        ha='center', fontsize=12, fontweight='bold', color='#16a085')

# Dataset
data = patches.FancyBboxPatch((2, 1.5), 6, 1.2,
                               boxstyle="round,pad=0.05",
                               edgecolor='#8e44ad', 
                               facecolor='#f4ecf7',
                               linewidth=3)
ax.add_patch(data)
ax.text(5, 2.35, 'Training Dataset (Distributed)', 
        ha='center', fontsize=16, fontweight='bold', color='#8e44ad')
ax.text(5, 1.9, 'WikiText-2 • Automatically Sharded Across GPUs', 
        ha='center', fontsize=11, style='italic')

# Arrows from data to GPUs
for i in range(4):
    x = 0.5 + i * 2.3 + gpu_width/2
    ax.arrow(x, 2.7, 0, 2.1,
             head_width=0.12, head_length=0.1, fc='#95a5a6', ec='#95a5a6', 
             linewidth=2, alpha=0.6)

# Performance box
perf_box = patches.FancyBboxPatch((0.3, 0.2), 3.5, 1.0,
                                   boxstyle="round,pad=0.05",
                                   edgecolor='#c0392b',
                                   facecolor='#fadbd8',
                                   linewidth=2)
ax.add_patch(perf_box)
ax.text(2.05, 0.95, '📊 Performance', ha='center', fontsize=12, fontweight='bold')
ax.text(2.05, 0.65, '• Speedup: 3.50x', ha='center', fontsize=10)
ax.text(2.05, 0.40, '• Efficiency: 87.5%', ha='center', fontsize=10)

# Tech stack box
tech_box = patches.FancyBboxPatch((6.2, 0.2), 3.5, 1.0,
                                   boxstyle="round,pad=0.05",
                                   edgecolor='#27ae60',
                                   facecolor='#d5f4e6',
                                   linewidth=2)
ax.add_patch(tech_box)
ax.text(7.95, 0.95, '🛠️ Stack', ha='center', fontsize=12, fontweight='bold')
ax.text(7.95, 0.65, '• PyTorch 2.7 + CUDA', ha='center', fontsize=10)
ax.text(7.95, 0.40, '• NCCL Backend', ha='center', fontsize=10)

plt.savefig('docs/images/architecture_diagram.png', dpi=300, bbox_inches='tight')
print("✅ Architecture diagram saved to: docs/images/architecture_diagram.png")

try:
    plt.show()
except:
    pass
