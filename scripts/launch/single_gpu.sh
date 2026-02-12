#!/bin/bash

echo "🚀 Launching single GPU training..."

python train.py \
    --config configs/single_gpu/train_tiny.yaml \
    --output_dir experiments/single_gpu_run

echo "✅ Training complete!"
