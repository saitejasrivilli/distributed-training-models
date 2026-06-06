import os
import socket
import torch
import torch.distributed as dist


def setup_distributed():
    """
    Initialize the process group for single-node or multi-node training.

    Reads the standard torchrun environment variables:
      RANK          — global rank across all nodes (0 … world_size-1)
      LOCAL_RANK    — rank within this node (0 … gpus_per_node-1)
      WORLD_SIZE    — total number of processes
      NODE_RANK     — index of this node (informational, not used by torch.dist)
      MASTER_ADDR   — hostname/IP of rank-0 process
      MASTER_PORT   — TCP port for NCCL rendezvous

    NCCL tuning applied automatically:
      - expandable_segments prevents CUDA allocator fragmentation
      - timeout set via NCCL_TIMEOUT env var (default 30 min)
    """
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        # Single-process fallback (no torchrun)
        return 0, 1, 0

    rank       = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    node_rank  = int(os.environ.get("NODE_RANK", 0))
    num_nodes  = world_size // max(1, torch.cuda.device_count())

    torch.cuda.set_device(local_rank)

    # expandable_segments: reduces peak VRAM from allocator fragmentation
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=__import__("datetime").timedelta(
            minutes=int(os.environ.get("NCCL_TIMEOUT_MIN", 30))
        ),
    )

    if rank == 0:
        print(
            f"[dist] world_size={world_size} | "
            f"nodes={num_nodes} | gpus_per_node={world_size // max(num_nodes, 1)} | "
            f"master={os.environ.get('MASTER_ADDR', 'localhost')}"
            f":{os.environ.get('MASTER_PORT', '29500')}",
            flush=True,
        )

    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def get_node_info():
    """Return (node_rank, num_nodes, local_rank, gpus_per_node) from env."""
    world_size    = int(os.environ.get("WORLD_SIZE", 1))
    local_rank    = int(os.environ.get("LOCAL_RANK", 0))
    node_rank     = int(os.environ.get("NODE_RANK", 0))
    gpus_per_node = torch.cuda.device_count()
    num_nodes     = max(1, world_size // max(1, gpus_per_node))
    return node_rank, num_nodes, local_rank, gpus_per_node
