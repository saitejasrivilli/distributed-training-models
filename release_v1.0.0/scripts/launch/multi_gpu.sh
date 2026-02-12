#!/bin/bash

# Configuration
NUM_GPUS=${1:-$(nvidia-smi --list-gpus | wc -l)}
CONFIG=${2:-"configs/data_parallel/train_117M.yaml"}
OUTPUT_DIR=${3:-"experiments/multi_gpu_run"}

echo "🚀 Launching ${NUM_GPUS}-GPU training..."
echo "Config: ${CONFIG}"
echo "Output: ${OUTPUT_DIR}"

torchrun \
    --standalone \
    --nproc_per_node=${NUM_GPUS} \
    train.py \
    --config ${CONFIG} \
    --output_dir ${OUTPUT_DIR}

echo "✅ Training complete!"
