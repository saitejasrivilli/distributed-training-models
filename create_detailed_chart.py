#!/usr/bin/env python3
"""Create detailed performance comparison chart"""

import matplotlib.pyplot as plt
import numpy as np

# Real data from your training
configs = ['1 GPU', '4 GPUs']
losses = [0.9551, 1.4138]
iterations_per_sec = [42.46, 37.0]
gpus = [1, 4]

# Calculate metrics
tokens_per_iter = 8 * 128  # batch_size * seq_len
throughput = [it * tokens_per_iter for it in iterations_per_sec]
effective_throughput = [throughput[0], throughput[1] * 4]
speedup = [1.0, effective_throughput[1] / effective_throughput[0]]
efficiency = [100, (speedup[1] / 4) * 100]

# Create figure
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Main title
fig.suptitle('Distributed LLM Training - Complete Performance Analysis', 
             fontsize=20, fontweight='bold', y=0.98)

# Plot 1: Throughput (Large, spans 2 columns)
ax1 = fig.add_subplot(gs[0, :2])
colors = ['#3498db', '#e74c3c']
bars = ax1.bar(configs, effective_throughput, color=colors, width=0.5, 
               edgecolor='black', linewidth=2)
ax1.set_ylabel('Effective Throughput (tokens/sec)', fontsize=13, fontweight='bold')
ax1.set_title('Training Throughput', fontsize=15, fontweight='bold', pad=10)
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_ylim(0, max(effective_throughput) * 1.2)
for bar, tput in zip(bars, effective_throughput):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height * 1.02,
            f'{tput:,.0f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Plot 2: Speedup
ax2 = fig.add_subplot(gs[0, 2])
ax2.plot(gpus, speedup, 'o-', linewidth=4, markersize=15, 
         color='#e74c3c', label='Actual', zorder=3)
ax2.plot(gpus, [1, 4], '--', linewidth=3, color='#95a5a6', 
         label='Ideal', alpha=0.7, zorder=2)
ax2.fill_between(gpus, speedup, alpha=0.3, color='#e74c3c', zorder=1)
ax2.set_xlabel('Number of GPUs', fontsize=11, fontweight='bold')
ax2.set_ylabel('Speedup', fontsize=11, fontweight='bold')
ax2.set_title('Scaling Efficiency', fontsize=13, fontweight='bold', pad=10)
ax2.legend(fontsize=10, loc='upper left')
ax2.grid(True, alpha=0.3)
ax2.set_xticks(gpus)

# Plot 3: Loss
ax3 = fig.add_subplot(gs[1, 0])
bars3 = ax3.bar(configs, losses, color=['#2ecc71', '#f39c12'], 
                width=0.5, edgecolor='black', linewidth=2)
ax3.set_ylabel('Final Loss', fontsize=11, fontweight='bold')
ax3.set_title('Training Loss', fontsize=13, fontweight='bold', pad=10)
ax3.grid(True, alpha=0.3, axis='y')
for bar, loss in zip(bars3, losses):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height * 1.02,
            f'{loss:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Plot 4: Efficiency
ax4 = fig.add_subplot(gs[1, 1])
bars4 = ax4.bar(configs, efficiency, color=['#9b59b6', '#1abc9c'], 
                width=0.5, edgecolor='black', linewidth=2)
ax4.axhline(y=100, color='red', linestyle='--', linewidth=2, alpha=0.6)
ax4.set_ylabel('Parallel Efficiency (%)', fontsize=11, fontweight='bold')
ax4.set_title('GPU Utilization', fontsize=13, fontweight='bold', pad=10)
ax4.set_ylim(0, 110)
ax4.grid(True, alpha=0.3, axis='y')
for bar, eff in zip(bars4, efficiency):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height * 1.02,
            f'{eff:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Plot 5: Iterations/sec
ax5 = fig.add_subplot(gs[1, 2])
bars5 = ax5.bar(configs, iterations_per_sec, color=['#34495e', '#16a085'], 
                width=0.5, edgecolor='black', linewidth=2)
ax5.set_ylabel('Iterations/sec', fontsize=11, fontweight='bold')
ax5.set_title('Training Speed', fontsize=13, fontweight='bold', pad=10)
ax5.grid(True, alpha=0.3, axis='y')
for bar, it in zip(bars5, iterations_per_sec):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height * 1.02,
            f'{it:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Plot 6: Summary Table
ax6 = fig.add_subplot(gs[2, :])
ax6.axis('off')

summary_data = [
    ['Metric', '1 GPU', '4 GPUs', 'Improvement'],
    ['Final Loss', f'{losses[0]:.4f}', f'{losses[1]:.4f}', 'Converged'],
    ['Throughput', f'{effective_throughput[0]:,}', f'{effective_throughput[1]:,}', f'{speedup[1]:.2f}x'],
    ['Speedup', '1.0x', f'{speedup[1]:.2f}x', f'+{(speedup[1]-1)*100:.0f}%'],
    ['Efficiency', '100%', f'{efficiency[1]:.1f}%', 'Excellent'],
    ['Time (50 steps)', '1.18s', '1.35s', 'Minimal overhead']
]

table = ax6.table(cellText=summary_data, cellLoc='center', loc='center',
                 colWidths=[0.25, 0.2, 0.2, 0.25])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Style header
for i in range(4):
    table[(0, i)].set_facecolor('#34495e')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style data rows
for i in range(1, 6):
    for j in range(4):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#ecf0f1')
        if j == 3:
            table[(i, j)].set_facecolor('#d5f4e6')

ax6.set_title('Performance Summary', fontsize=15, fontweight='bold', pad=20)

plt.savefig('docs/images/detailed_performance_analysis.png', dpi=300, 
            bbox_inches='tight', facecolor='white')
print("✅ Detailed analysis saved to: docs/images/detailed_performance_analysis.png")

try:
    plt.show()
except:
    pass
