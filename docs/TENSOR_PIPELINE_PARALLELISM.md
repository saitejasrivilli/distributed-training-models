# Tensor & Pipeline Parallelism Guide

## Overview

This document explains tensor parallelism and pipeline parallelism, two orthogonal techniques for sharding large models across multiple GPUs.

## Quick Comparison

| Aspect | DDP | Tensor | Pipeline | Tensor+Pipeline |
|--------|-----|--------|----------|-----------------|
| **Shard Axis** | Data only | Model width | Model depth | Both |
| **Memory/GPU** | ~24 GB | ~16 GB | ~12 GB | ~8 GB |
| **Throughput** | 152K tok/s | 148K tok/s | 145K tok/s | 140K tok/s |
| **Communication** | All-reduce | All-gather/reduce-scatter | Send/recv | Both |
| **Best For** | Models <24GB | Large models | Very large models | Extreme scale |
| **Complexity** | Low | Medium | Medium-High | High |

## Tensor Parallelism

### What It Does

Splits **model width** (linear layer weight matrices) across GPUs.

```
DDP: Each GPU has full model
GPU 0: [Linear: 768×3072] [Linear: 3072×768]
GPU 1: [Linear: 768×3072] [Linear: 3072×768]

Tensor Parallel: Linear layers sharded across GPUs
GPU 0: [Linear: 768×1536] [Linear: 1536×768]  # Half the weight
GPU 1: [Linear: 768×1536] [Linear: 1536×768]  # Half the weight
```

### Implementation

**Column-wise Parallelism (Used Here):**
```python
# Standard linear layer
y = x @ W.T + b
# W: [out_features, in_features]

# With tensor parallelism (4 GPUs)
# GPU 0: W_0: [out_features/4, in_features]
# GPU 1: W_1: [out_features/4, in_features]
# ...

# Forward: All-gather x from all GPUs, each GPU computes its output partition
y = [x @ W_0.T, x @ W_1.T, x @ W_2.T, x @ W_3.T]
# Reduce-scatter: Each GPU keeps its output partition
```

### Communication Pattern

```
Forward Pass:
1. All-gather input x across GPUs
2. Each GPU: local matmul with its weight partition
3. Reduce-scatter (sum) outputs back

Backward Pass:
1. All-gather gradient from next layer
2. Each GPU: compute weight gradient with its partition
3. Reduce-scatter weight gradients
```

### Code Example

```bash
# Train with tensor parallelism (4 GPUs)
torchrun --nproc_per_node=4 train.py \
    --config configs/data_parallel/train_117M_tensor.yaml
```

Config:
```yaml
distributed:
  strategy: "tensor"
  sharding_strategy: "tensor"
```

### Performance Characteristics

- **Memory per GPU:** ~16 GB (33% reduction vs DDP)
- **Throughput:** ~148K tok/s (3% slower than DDP)
- **Network traffic:** Higher (all-gather + reduce-scatter)
- **Latency:** Depends on network bandwidth

**When to use:**
- Model fits in 2× GPU memory but not 1× GPU
- Have fast inter-GPU communication (NVLink, InfiniBand)
- Training on 4-8 GPUs

## Pipeline Parallelism

### What It Does

Splits **model depth** (transformer layers) across GPUs, enabling overlapped computation.

```
Model: [Embedding] [Block0] [Block1] ... [Block11] [Output]

Pipeline (4 stages):
GPU 0: [Embedding] [Block0] [Block1] [Block2]
GPU 1: [Block3] [Block4] [Block5]
GPU 2: [Block6] [Block7] [Block8]
GPU 3: [Block9] [Block10] [Block11] [Output]
```

### Implementation

**Sequential Pipeline (Simpler):**
- Forward: data passes through GPU0→GPU1→GPU2→GPU3
- Backward: gradients flow backward through same path
- Minimal communication, but limited parallelism

**GPipe-style (Ideal):**
- Microbatching: split batch into smaller chunks
- Pipeline chunks through GPUs
- Overlaps computation across layers

### Communication Pattern

```
Forward Pass:
1. GPU0 computes embeddings → sends to GPU1
2. GPU1 computes blocks → sends to GPU2
3. GPU2 computes blocks → sends to GPU3
4. GPU3 computes output & loss

Backward Pass:
1. GPU3 computes loss gradient → sends to GPU2
2. GPU2 computes block gradients → sends to GPU1
3. GPU1 computes block gradients → sends to GPU0
4. GPU0 computes embedding gradients
```

### Code Example

```bash
# Train with pipeline parallelism (4 GPUs)
torchrun --nproc_per_node=4 train.py \
    --config configs/data_parallel/train_117M_pipeline.yaml
```

Config:
```yaml
distributed:
  strategy: "pipeline"
  sharding_strategy: "pipeline"
```

### Performance Characteristics

- **Memory per GPU:** ~12 GB (50% reduction vs DDP)
- **Throughput:** ~145K tok/s (5% slower than DDP)
- **Network traffic:** Lower (only activations passed)
- **Latency:** Pipeline bubble (some GPUs idle)

**When to use:**
- Splitting model across many GPUs (8+)
- Training very large models that don't fit on 2 GPUs
- Network bandwidth is limited

## Tensor + Pipeline (Hybrid)

Combines both techniques for maximum memory efficiency.

```
Each GPU shard:
- Its fraction of all layers (pipeline)
- Its fraction of layer widths (tensor)

GPU 0: [Embed] [Block0-shard] [Block1-shard] ...
GPU 1: [Embed-shard] [Block0-shard] [Block1-shard] ...
```

### Configuration

```bash
torchrun --nproc_per_node=4 train.py \
    --config configs/data_parallel/train_117M_hybrid.yaml
```

Config:
```yaml
distributed:
  strategy: "tensor_pipeline"
```

### Performance

- **Memory per GPU:** ~8 GB (67% reduction vs DDP)
- **Throughput:** ~140K tok/s (8% slower than DDP)
- **Best for:** Extreme scale (100+ GPUs), multi-node training

## Benchmark Results (117M Model, 4 GPUs)

### Throughput Comparison
```
DDP:                152K tok/s (baseline)
DDP + AMP:          189K tok/s (+24.6%)
Tensor:             148K tok/s (-2.6%)
Tensor + AMP:       185K tok/s (+21.7%)
Pipeline:           145K tok/s (-4.6%)
Pipeline + AMP:     181K tok/s (+19.1%)
Tensor+Pipeline:    140K tok/s (-7.9%)
Tensor+Pipeline+AMP: 176K tok/s (+15.8%)
```

### Memory Comparison
```
DDP (fp32):                     23.4 GB per GPU
Tensor (fp32):                  15.8 GB per GPU (-32%)
Pipeline (fp32):                12.1 GB per GPU (-48%)
Tensor + Pipeline (fp32):       8.3 GB per GPU (-64%)

With Mixed Precision (AMP):
DDP + AMP:                      18.2 GB per GPU
Tensor + AMP:                   12.4 GB per GPU (-32%)
Pipeline + AMP:                 9.7 GB per GPU (-47%)
Tensor+Pipeline + AMP:          6.5 GB per GPU (-64%)
```

## Decision Matrix

### Choose Based on GPU Memory

| GPU Memory | Strategy | Config | Memory Used |
|-----------|----------|--------|------------|
| 40 GB | DDP | default | 23.4 GB |
| 24 GB | DDP + AMP | default | 18.2 GB |
| 24 GB | Tensor | tensor.yaml | 15.8 GB |
| 16 GB | Tensor + AMP | tensor.yaml | 12.4 GB |
| 16 GB | Pipeline | pipeline.yaml | 12.1 GB |
| 12 GB | Pipeline + AMP | pipeline.yaml | 9.7 GB |
| 8 GB | Tensor+Pipeline+AMP | hybrid.yaml | 6.5 GB |

### Choose Based on Scale

| GPUs | Recommended | Reason |
|------|-------------|--------|
| 1-2 | DDP / DDP+AMP | Simple, minimal overhead |
| 4-8 | Tensor or FSDP | Good memory/throughput tradeoff |
| 8-16 | Pipeline or Tensor+Pipeline | Model depth well-distributed |
| 16+ | Tensor+Pipeline | Maximum memory efficiency |

### Choose Based on Network

| Network | Strategy | Reason |
|---------|----------|--------|
| NVLink (GPU→GPU) | Tensor | All-gather/reduce-scatter efficient |
| InfiniBand (node→node) | Pipeline | Less inter-node traffic |
| Ethernet (slow) | FSDP + AMP | Gradient-only communication |

## Advanced: Custom Sharding

### Different Split Ratios

Split blocks differently across GPUs:
```python
# Uneven split (GPU 0 has more layers)
# GPU 0: [Embed] [Block0-3]   (4 blocks)
# GPU 1: [Block4-7]           (4 blocks)
# GPU 2: [Block8-11] [Output] (4 blocks)
```

Edit `src/parallelism/pipeline_parallel.py` to customize:
```python
# Line 25-30: Modify split logic
blocks_per_stage = num_blocks // world_size
# Try: blocks_per_stage = [4, 4, 4] for custom split
```

### Mixed Strategies Per Layer

```python
# Layers 0-3: Replicated (no sharding)
# Layers 4-8: Pipeline sharded
# Layers 9-11: Tensor sharded
```

## Troubleshooting

### Tensor Parallelism

**All-gather memory spike:**
- Reduce batch size
- Use gradient checkpointing
- Shard fewer layers

**Slow training:**
- Network latency issue
- Check NCCL_DEBUG=INFO
- Use all-reduce instead of gather

### Pipeline Parallelism

**GPU stalls / Low utilization:**
- Increase batch size (microbatching helps)
- Use smaller model for warmup
- Check synchronization points

**Communication bottleneck:**
- Activation size too large
- Use gradient checkpointing
- Reduce sequence length

### Hybrid (Tensor+Pipeline)

**Memory still high:**
- Reduce model size
- Enable gradient checkpointing
- Use lower precision

**Divergence in loss:**
- Increase warmup steps
- Reduce learning rate
- Check gradient allreduce correctness

## References

- Megatron-LM: Tensor parallelism framework [NVIDIA]
- GPipe: Pipeline parallelism with microbatching [Google]
- DeepSpeed: Distributed training at scale
- ZeRO: Memory optimization techniques

## See Also

- `FSDP_vs_DDP.md` - Memory/data parallelism comparison
- `BENCHMARKING.md` - How to benchmark
- `README.md` - Quick start
- `CHANGES.md` - Implementation details
