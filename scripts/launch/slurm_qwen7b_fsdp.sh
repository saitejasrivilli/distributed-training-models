#!/bin/bash
#SBATCH --job-name=qwen7b-fsdp
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=experiments/logs/slurm-qwen7b-%j.out
#SBATCH --error=experiments/logs/slurm-qwen7b-%j.err
#SBATCH --partition=gpu

# ── Environment ──────────────────────────────────────────────────────────────
module load cuda/12.1 python/3.10 2>/dev/null || true
source venv/bin/activate 2>/dev/null || true

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29600
NNODES=$SLURM_NNODES
GPUS_PER_NODE=$SLURM_GPUS_PER_NODE
WORLD_SIZE=$((NNODES * GPUS_PER_NODE))

echo "===== Qwen2.5-7B FSDP Post-Training ====="
echo "Job ID:     $SLURM_JOB_ID"
echo "Nodes:      $NNODES   World size: $WORLD_SIZE"
echo "Master:     $MASTER_ADDR:$MASTER_PORT"
echo "Memory per GPU target: ~10 GB (FULL_SHARD on $WORLD_SIZE GPUs)"
echo "=========================================="

export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=ib0
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=3600
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/scratch/$USER/hf_cache
export TRANSFORMERS_CACHE=/scratch/$USER/hf_cache

srun bash -c "
  torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --node_rank=\$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train.py \
      --config configs/multi_node/qwen7b_fsdp.yaml \
      --output_dir experiments/qwen7b_fsdp_${SLURM_JOB_ID}
"

echo "Qwen2.5-7B FSDP training complete"
