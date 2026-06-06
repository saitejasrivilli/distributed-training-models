#!/bin/bash
#SBATCH --job-name=fsdp-2node
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1        # one torchrun launcher per node
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=experiments/logs/slurm-%j.out
#SBATCH --error=experiments/logs/slurm-%j.err
#SBATCH --partition=gpu            # adjust to your cluster partition name

# ── Environment ──────────────────────────────────────────────────────────────
module load cuda/12.1 python/3.10 2>/dev/null || true
source venv/bin/activate 2>/dev/null || true

# ── Distributed rendezvous ────────────────────────────────────────────────────
# SLURM allocates the node list; we take the first node as master.
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500
NNODES=$SLURM_NNODES
GPUS_PER_NODE=$SLURM_GPUS_PER_NODE
WORLD_SIZE=$((NNODES * GPUS_PER_NODE))

echo "===== SLURM Multi-Node FSDP ====="
echo "Job ID:        $SLURM_JOB_ID"
echo "Nodes:         $NNODES ($SLURM_JOB_NODELIST)"
echo "GPUs/node:     $GPUS_PER_NODE"
echo "World size:    $WORLD_SIZE"
echo "Master:        $MASTER_ADDR:$MASTER_PORT"
echo "================================="

# ── NCCL settings ─────────────────────────────────────────────────────────────
# Use InfiniBand if available, fall back to Ethernet.
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=ib0          # change to eth0 if no IB
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=1800               # 30 min — needed for large AllReduce

# Prevent CUDA OOM from fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Launch one torchrun per node via srun ────────────────────────────────────
# srun distributes this command across all allocated nodes.
# NODE_RANK is set per-node by SLURM through $SLURM_NODEID.
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
      --config configs/multi_node/fsdp_2node_8gpu.yaml \
      --output_dir experiments/fsdp_2node_${SLURM_JOB_ID}
"

echo "Training complete — job $SLURM_JOB_ID"
