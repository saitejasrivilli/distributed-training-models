#!/usr/bin/env python3
"""Track training metrics in real-time"""

import re
import json
from pathlib import Path
import matplotlib.pyplot as plt

def parse_training_log(log_file):
    """Parse training output to extract metrics"""
    
    steps = []
    losses = []
    
    # Look for patterns like "50/50 [00:01<00:00, 47.65it/s, loss=2.1757]"
    pattern = r'(\d+)/\d+.*loss=([\d.]+)'
    
    with open(log_file, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                step = int(match.group(1))
                loss = float(match.group(2))
                steps.append(step)
                losses.append(loss)
    
    return steps, losses

def create_live_tracker(output_dir):
    """Create a live training tracker"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tracker_file = output_dir / 'training_metrics.json'
    
    return {
        'steps': [],
        'losses': [],
        'file': tracker_file
    }

if __name__ == "__main__":
    print("Training tracker created!")
