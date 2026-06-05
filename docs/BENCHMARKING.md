# Benchmarking Guide: DDP vs FSDP Performance Comparison

## Overview

This guide shows how to run comprehensive benchmarks comparing different distributed training configurations: DDP with/without mixed precision and FSDP with/without mixed precision.

## Quick Start

Run benchmarks on your hardware (4 GPUs):

```bash
# Basic benchmark (100 steps)
torchrun --nproc_per_node=4 benchmark_training.py \
    --config configs/data_parallel/train_117M.yaml \
    --num_steps 100 \
    --output_dir benchmarks

# Extended benchmark (500 steps for stable metrics)
torchrun --nproc_per_node=4 benchmark_training.py \
    --config configs/data_parallel/train_117M.yaml \
    --num_steps 500 \
    --output_dir my_benchmarks
```

Results are saved to: `benchmarks/benchmark_results.json`

## Understanding Benchmark Results

### Output Example

```
============================================================
🔬 Training Configuration Benchmarks
============================================================

Benchmarking: DDP without mixed precision
  ✓ Time: 45.23s
  ✓ Throughput: 152142 samples/sec
  ✓ Memory allocated: 23.40 GB

Benchmarking: DDP with mixed precision (fp16)
  ✓ Time: 37.85s
  ✓ Throughput: 189534 samples/sec
  ✓ Memory allocated: 18.20 GB

...

============================================================
📊 Benchmark Results Summary
============================================================

ddp_amp0:
  Throughput: 152142 samples/sec
  Memory: 23.40 GB
  Time: 45.23s

ddp_amp1:
  Throughput: 189534 samples/sec
  Relative speedup: 1.246x
  Memory: 18.20 GB
  Time: 37.85s

fsdp_amp0:
  Throughput: 148900 samples/sec
  Relative speedup: 0.979x
  Memory: 14.20 GB
  Time: 46.12s

fsdp_amp1:
  Throughput: 185200 samples/sec
  Relative speedup: 1.218x
  Memory: 11.80 GB
  Time: 41.03s

✅ Results saved to benchmarks/benchmark_results.json
```

### Interpreting Results

#### Throughput (samples/sec)
- **Higher is better** for training speed
- Measured as (num_steps × batch_size) / elapsed_time
- Affected by GPU type, batch size, and model size

#### Relative Speedup
- Compared to baseline (DDP without AMP)
- Example: `1.246x` means 24.6% faster than baseline

#### Memory Allocated
- GPU memory used during training
- Important for determining which config fits on your hardware
- Includes model, optimizer, activations, and gradients

#### Time
- Wall-clock time for benchmark (seconds)
- Includes 5 warmup steps + benchmark steps
- Longer benchmarks are more stable

## Configuration Guide

### Using Different Configs

#### Standard 117M Model (Recommended for Testing)
```bash
torchrun --nproc_per_node=4 benchmark_training.py \
    --config configs/data_parallel/train_117M.yaml \
    --num_steps 100
```

#### Tiny Model (Fast Testing)
```bash
torchrun --nproc_per_node=4 benchmark_training.py \
    --config configs/single_gpu/train_tiny.yaml \
    --num_steps 50
```

### Custom Configurations

Create your own config file:
```yaml
# custom_benchmark.yaml
model:
  name: "medium"  # Larger model for more realistic benchmark

training:
  batch_size: 1024  # Larger batch
  mixed_precision_training: true  # Enable AMP

distributed:
  strategy: "ddp"  # or "fsdp"
  backend: "nccl"

# ... rest of config ...
```

Then run:
```bash
torchrun --nproc_per_node=4 benchmark_training.py \
    --config custom_benchmark.yaml \
    --num_steps 100
```

## Running Benchmarks on Different Hardware

### Single GPU (for comparison)
```bash
python benchmark_training.py \
    --config configs/single_gpu/train_tiny.yaml \
    --num_steps 50
```

### 2 GPUs
```bash
torchrun --nproc_per_node=2 benchmark_training.py \
    --config configs/data_parallel/train_117M.yaml \
    --num_steps 100
```

### 8 GPUs
```bash
torchrun --nproc_per_node=8 benchmark_training.py \
    --config configs/data_parallel/train_117M.yaml \
    --num_steps 100
```

### Multi-Node (8 GPUs across 2 nodes)
```bash
# On node 0:
torchrun --nproc_per_node=4 \
         --nnodes=2 \
         --node_rank=0 \
         --master_addr=<node0_ip> \
         --master_port=29500 \
         benchmark_training.py \
    --config configs/data_parallel/train_117M.yaml \
    --num_steps 100

# On node 1:
torchrun --nproc_per_node=4 \
         --nnodes=2 \
         --node_rank=1 \
         --master_addr=<node0_ip> \
         --master_port=29500 \
         benchmark_training.py \
    --config configs/data_parallel/train_117M.yaml \
    --num_steps 100
```

## Analyzing Results

### JSON Output Format

```json
{
  "ddp_amp0": {
    "elapsed_time": 45.23,
    "throughput_samples_per_sec": 152142,
    "allocated_memory_gb": 23.40,
    "reserved_memory_gb": 24.12,
    "strategy": "ddp",
    "use_amp": false,
    "num_steps": 100,
    "relative_throughput": 1.0
  },
  "ddp_amp1": {
    "elapsed_time": 37.85,
    "throughput_samples_per_sec": 189534,
    "allocated_memory_gb": 18.20,
    "reserved_memory_gb": 19.05,
    "strategy": "ddp",
    "use_amp": true,
    "num_steps": 100,
    "relative_throughput": 1.246
  },
  ...
}
```

### Processing Results with Python

```python
import json

# Load results
with open('benchmarks/benchmark_results.json') as f:
    results = json.load(f)

# Find best throughput
best = max(results.items(), key=lambda x: x[1]['throughput_samples_per_sec'])
print(f"Best throughput: {best[0]} with {best[1]['throughput_samples_per_sec']:.0f} tok/s")

# Find best memory efficiency
smallest_memory = min(results.items(), key=lambda x: x[1]['allocated_memory_gb'])
print(f"Smallest memory: {smallest_memory[0]} with {smallest_memory[1]['allocated_memory_gb']:.2f} GB")

# Memory savings with FSDP
ddp_baseline = results['ddp_amp0']['allocated_memory_gb']
for key, val in results.items():
    saving = (1 - val['allocated_memory_gb'] / ddp_baseline) * 100
    print(f"{key}: {saving:.1f}% memory saved vs DDP baseline")
```

## Performance Expectations

### Typical Results (4x NVIDIA A100 80GB)

| Config | Throughput | Memory | Relative Speed |
|--------|-----------|--------|-----------------|
| DDP fp32 | 152K tok/s | 23.4 GB | 1.00x |
| DDP fp16 | 189K tok/s | 18.2 GB | 1.25x |
| FSDP fp32 | 149K tok/s | 14.2 GB | 0.98x |
| FSDP fp16 | 185K tok/s | 11.8 GB | 1.22x |

Your results may vary based on:
- GPU type (A100 vs H100 vs V100)
- GPU memory capacity
- Network bandwidth
- Model size
- Batch size

## Optimization Tips

### To Maximize Throughput
1. Use DDP + Mixed Precision
2. Increase batch size (if memory allows)
3. Disable gradient checkpointing if not needed
4. Use flash attention if available

### To Minimize Memory
1. Use FSDP + Mixed Precision
2. Reduce batch size
3. Enable gradient checkpointing
4. Use lower precision (fp16 vs fp32)

### For Stable Benchmarks
1. Run more steps (500+ for stable metrics)
2. Allow warm-up steps
3. Run multiple times and average
4. Check CPU/network utilization is not saturated

## Troubleshooting

### Out of Memory Error
- Reduce batch size in config
- Use FSDP instead of DDP
- Enable mixed precision training
- Try gradient checkpointing

### Slow Benchmark
- Check for other processes using GPU
- Verify NCCL backend is available
- Run on fewer GPUs first
- Check network bandwidth (for multi-node)

### Missing Results File
- Check `--output_dir` is writable
- Ensure rank 0 process completed
- Look for errors in console output
- Try with explicit directory: `--output_dir /tmp/benchmarks`

## Advanced Benchmarking

### Customizing Benchmark Script

Edit `benchmark_training.py` to:
- Add new strategies (SHARD_GRAD_OP, NO_SHARD)
- Test different batch sizes
- Add gradient checkpointing
- Measure convergence metrics
- Track per-layer memory usage

Example: Test all FSDP strategies
```python
configs_to_test = [
    ('ddp', False, 'DDP baseline'),
    ('fsdp', False, 'FSDP FULL_SHARD'),
    ('fsdp_grad', False, 'FSDP SHARD_GRAD_OP'),
    ('fsdp_no', False, 'FSDP NO_SHARD'),
]
```

## See Also

- `FSDP_vs_DDP.md` - Detailed comparison and guidance
- `README.md` - Quick start guide
- `TRAINING.md` - Training instructions
- `CHANGES.md` - Implementation details
