#!/bin/bash
# Manual multi-node torchrun launch — no SLURM required.
#
# Run this script on EVERY node, changing NODE_RANK per node.
# NODE0 runs with NODE_RANK=0 (master); all others follow.
#
# Usage:
#   # On node0 (master):
#   MASTER_ADDR=192.168.1.10 NODE_RANK=0 NNODES=2 bash torchrun_multi_node.sh
#
#   # On node1:
#   MASTER_ADDR=192.168.1.10 NODE_RANK=1 NNODES=2 bash torchrun_multi_node.sh
#
# Environment variables (all required on each node):
#   MASTER_ADDR  — IP or hostname of node0 (must be reachable from all nodes)
#   NODE_RANK    — this node's rank: 0, 1, 2, ...
#   NNODES       — total number of nodes (default: 2)

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"
NODE_RANK="${NODE_RANK:-0}"
NNODES="${NNODES:-2}"
GPUS_PER_NODE="${GPUS_PER_NODE:-$(nvidia-smi --list-gpus | wc -l)}"
CONFIG="${CONFIG:-configs/multi_node/fsdp_2node_8gpu.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/multi_node_$(date +%Y%m%d_%H%M%S)}"

WORLD_SIZE=$((NNODES * GPUS_PER_NODE))

echo "================================================="
echo "torchrun multi-node launch"
echo "  Master:      $MASTER_ADDR:$MASTER_PORT"
echo "  This node:   rank $NODE_RANK / $NNODES"
echo "  GPUs/node:   $GPUS_PER_NODE   World size: $WORLD_SIZE"
echo "  Config:      $CONFIG"
echo "  Output:      $OUTPUT_DIR"
echo "================================================="

# ── NCCL tuning ───────────────────────────────────────────────────────────────
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800}"
# Use InfiniBand if present, else Ethernet
if ls /dev/infiniband/uverbs* &>/dev/null 2>&1; then
    export NCCL_IB_DISABLE=0
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ib0}"
else
    export NCCL_IB_DISABLE=1
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Launch ────────────────────────────────────────────────────────────────────
# --rdzv_backend=c10d: static rendezvous — all nodes connect to master_addr.
# Alternative: --rdzv_backend=etcd for dynamic (requires etcd service).
torchrun \
    --nnodes="$NNODES" \
    --nproc_per_node="$GPUS_PER_NODE" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    --rdzv_id="fsdp_run_$(date +%s)" \
    train.py \
        --config "$CONFIG" \
        --output_dir "$OUTPUT_DIR"
