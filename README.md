# 🚀 Distributed LLM Pre-Training from Scratch

[![Live Demo](https://img.shields.io/badge/Demo-Live%20on%20Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://distributed-training-models.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.7](https://img.shields.io/badge/pytorch-2.7.1-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![AMP Speedup](https://img.shields.io/badge/AMP%20Speedup-3.14x%20single%20GPU-orange)](https://github.com/saitejasrivilli/distributed-training-models)
[![FSDP Memory](https://img.shields.io/badge/FSDP-up%20to%2021%25%20memory%20saved-brightgreen)](https://github.com/saitejasrivilli/distributed-training-models)
[![PP Memory](https://img.shields.io/badge/Pipeline%20Parallel-75%25%20memory%20per%20GPU-blue)](https://github.com/saitejasrivilli/distributed-training-models)

> **🎮 [Try the Interactive Demo →](https://distributed-training-models.streamlit.app/)** | All benchmark numbers are real — measured on 4× NVIDIA A30 GPUs

<p align="center">
  <a href="https://distributed-training-models.streamlit.app/"><img src="https://img.shields.io/badge/🎮_Try_Live_Demo-Click_Here-FF4B4B?style=for-the-badge" alt="Live Demo"></a>
</p>

---

## 🎯 Project Highlights

Production-grade distributed training system for GPT-style language models, with end-to-end benchmarks comparing DDP, FSDP, and mixed precision across 1–4 GPUs.

<table>
<tr>
<td>

### ⚡ Mixed Precision (AMP)
- **3.14× throughput** on single A30
- fp32 → fp16: **8,300 → 26,097 tok/s**
- **12% less GPU memory** (12% less activation footprint)

</td>
<td>

### 🔒 FSDP vs DDP
- Up to **21% less memory/GPU** (4-GPU)
- FSDP: **3.96 GB** vs DDP: **5.02 GB** (fp16)
- Optimizer states sharded across GPUs

</td>
<td>

### 🔬 Fully Measured
- **18 configurations** benchmarked
- DDP, FSDP, TP, PP × 1–4 GPU × fp32/fp16
- Raw results in `benchmarks/real_results.json`

</td>
</tr>
</table>

---

## 🎮 Interactive Features

Experience the system without any setup:

| Feature | Description | Link |
|---------|-------------|------|
| 📊 **Performance Dashboard** | Real-time metrics and visualizations | [View](https://distributed-training-models.streamlit.app/) |
| ⚙️ **Scaling Calculator** | Calculate training time & costs for your use case | [Calculate](https://distributed-training-models.streamlit.app/) |
| 🎯 **Training Visualizer** | See training curves and GPU metrics | [Visualize](https://distributed-training-models.streamlit.app/) |

---

## ⚡ Quick Start

```bash
# Clone the repository
git clone https://github.com/saitejasrivilli/distributed-training-models.git
cd distributed-training-models

# Install dependencies (CUDA 11.8)
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install datasets transformers pyyaml tqdm

# Quick sanity test (single GPU, tiny model, 50 steps)
python train.py --config configs/quick_test.yaml --output_dir experiments/quick_test

# Single GPU — 1000 steps
python train.py --config configs/single_gpu/train_tiny.yaml --output_dir experiments/single_gpu

# 4-GPU DDP
torchrun --nproc_per_node=4 train.py \
    --config configs/data_parallel/train_117M.yaml \
    --output_dir experiments/ddp_run

# 4-GPU FSDP (lower memory per GPU)
torchrun --nproc_per_node=4 train.py \
    --config configs/data_parallel/train_117M_fsdp.yaml \
    --output_dir experiments/fsdp_run

# Reproduce DDP/FSDP/AMP benchmark (1 GPU then 2/4 GPU)
python run_benchmark.py
torchrun --nproc_per_node=2 run_benchmark.py
torchrun --nproc_per_node=4 run_benchmark.py

# Reproduce Tensor Parallel + Pipeline Parallel benchmark
torchrun --nproc_per_node=2 run_tp_pp_benchmark.py
torchrun --nproc_per_node=4 run_tp_pp_benchmark.py
# → all 18 configs merged into benchmarks/real_results.json
```

---

## Tests

```bash
# Fast unit tests — no GPU cluster needed
pytest test_train.py tests/ -v
```

7 tests pass: model creation, forward/backward pass, CUDA availability, YAML config loading, parameter count accuracy (config formula matches actual model).

---

## 📊 Real Benchmark Results

> Measured on **NVIDIA A30 GPUs (PCIe)**, **125M-param GPT model**, batch=4/GPU, seq_len=512.  
> All numbers from `benchmarks/real_results.json`. No estimates.

### Option 2 — Mixed Precision Training (AMP)

fp16 AMP is the single highest-ROI optimization in this pipeline.

| Configuration | Throughput | Peak Mem/GPU | vs 1-GPU fp32 |
|---------------|-----------|--------------|---------------|
| **1 GPU — fp32 (baseline)** | 8,300 tok/s | 5.21 GB | 1.00× |
| **1 GPU — fp16 AMP** | **26,097 tok/s** | **4.56 GB** | **3.14× faster, −12% mem** |
| 2 GPU DDP — fp32 | 7,123 tok/s | 5.67 GB/GPU | 0.86× |
| 2 GPU DDP — fp16 AMP | 20,522 tok/s | 5.02 GB/GPU | 2.47× |
| 4 GPU DDP — fp32 | 10,921 tok/s | 5.67 GB/GPU | 1.32× |
| **4 GPU DDP — fp16 AMP** | **20,950 tok/s** | **5.02 GB/GPU** | **2.52×** |

**Key insight:** AMP delivers **3.14× throughput** on a single A30 GPU — more than doubling the benefit of adding a second GPU at this model scale. For a 125M model on PCIe-connected GPUs, AMP is the highest-ROI optimization.

### Option 1 — DDP vs FSDP: Memory & Throughput Tradeoff

FSDP shards model parameters, gradients, and optimizer states across GPUs. Each GPU holds 1/N of the full state at rest, gathering parameters on-demand during forward/backward passes.

| Strategy | GPUs | Throughput | Mem/GPU | Memory vs DDP | Throughput vs DDP |
|----------|------|-----------|---------|---------------|-------------------|
| DDP fp32 | 2 | 7,123 tok/s | 5.67 GB | baseline | baseline |
| **FSDP fp32** | 2 | 7,261 tok/s | **4.97 GB** | **−12.3%** | +1.9% |
| DDP fp16 | 2 | 20,522 tok/s | 5.02 GB | baseline | baseline |
| **FSDP fp16** | 2 | **21,844 tok/s** | **4.31 GB** | **−14.1%** | **+6.4%** |
| DDP fp32 | 4 | 10,921 tok/s | 5.67 GB | baseline | baseline |
| **FSDP fp32** | 4 | 10,387 tok/s | **4.63 GB** | **−18.3%** | −4.9% |
| DDP fp16 | 4 | 20,950 tok/s | 5.02 GB | baseline | baseline |
| **FSDP fp16** | 4 | 17,840 tok/s | **3.96 GB** | **−21.1%** | −14.8% |

**Key insights:**
- At 2 GPUs in fp16, **FSDP is actually faster than DDP** (+6.4%) — because smaller parameter shards reduce the volume of data synchronized per all-reduce
- At 4 GPUs, FSDP costs 5–15% throughput to save 18–21% memory — worthwhile when GPU memory is the scaling bottleneck
- Memory savings **grow with GPU count**: 12% at 2 GPU → 21% at 4 GPU, because each GPU holds 1/N of the sharded state

### DDP Scaling Efficiency

Honest numbers for PCIe-connected A30 GPUs at 125M params:

| GPUs | Strategy | Precision | Total Throughput | Speedup vs 1-GPU fp32 | Parallel Efficiency |
|------|----------|-----------|-----------------|----------------------|---------------------|
| 1 | — | fp32 | 8,300 tok/s | 1.00× | 100% |
| 1 | — | **fp16** | **26,097 tok/s** | **3.14×** | — (AMP, not scaling) |
| 2 | DDP | fp32 | 7,123 tok/s | 0.86× | 42.9% |
| 2 | DDP | fp16 | 20,522 tok/s | 2.47× | — |
| 4 | DDP | fp32 | 10,921 tok/s | 1.32× | 32.9% |
| 4 | DDP | fp16 | 20,950 tok/s | 2.52× | — |

**Why efficiency is low:** A30 GPUs here are PCIe-connected (no NVLink). Each DDP step all-reduces ~650 MB of fp32 gradients across GPUs. At 125M params, the compute-to-communication ratio is unfavorable — DDP efficiency typically improves significantly with larger models (>1B params) or NVLink interconnects, where compute dominates communication.

---

### Option 3 — Tensor Parallelism: Column-Wise Weight Sharding

Tensor parallelism splits each linear layer's weight matrix across GPUs along the output dimension. Each GPU holds `out_features / N` rows of every weight, computes a partial output, then all_gather reconstructs the full output before the next layer.

**Implementation:** `src/parallelism/tensor_parallel.py` — `ParallelLinear` replaces every `nn.Linear` in the model. The `lm_head` (vocab size 50257, not divisible by 2 or 4) is kept on each rank unchanged.

```python
from src.parallelism.tensor_parallel import convert_linear_to_tensor_parallel

# Convert all linear layers to column-parallel (skip lm_head)
model = GPTModel(config).to(device)
model = convert_linear_to_tensor_parallel(
    model, world_size=world_size, rank=rank, skip_modules={'lm_head'}
)
# Each GPU now holds 1/N of every weight matrix row.
# Forward: local matmul → all_gather → full output (gradient-aware via custom autograd Function)
```

| GPUs | Precision | Throughput | Mem/GPU | vs 1-GPU same precision |
|------|-----------|-----------|---------|------------------------|
| 1 | fp32 | 8,300 tok/s | 5.21 GB | 1.00× (baseline) |
| 2 | fp32 | 4,387 tok/s | 4.77 GB | **0.53×** |
| 4 | fp32 | 2,258 tok/s | 4.49 GB | **0.27×** |
| 1 | fp16 | 26,097 tok/s | 4.56 GB | 1.00× (baseline) |
| 2 | fp16 | 10,476 tok/s | 4.04 GB | **0.40×** |
| 4 | fp16 | 3,820 tok/s | 3.72 GB | **0.15×** |

**Why throughput drops with more GPUs:** The 125M-param GPT-small has 12 blocks × 4 linear layers = **48 all_gather operations per forward pass** (plus 48 all_reduce on input gradients in backward). On PCIe (effective bandwidth ~32 GB/s bidirectional), communication dominates computation at this model scale.

**What TP is actually for:** Tensor parallelism is designed for models whose individual layers are too large to fit on a single GPU — e.g., a single attention projection of width 8192 in a 70B model. At that scale, each GPU holds 1/N of a massive weight matrix, and NVLink bandwidth (600+ GB/s) makes the all_gather fast relative to the matmul. On PCIe with a 125M model, TP is communication-bound.

---

### Option 4 — Pipeline Parallelism: Layer-Wise Model Sharding

Pipeline parallelism assigns a contiguous slice of transformer blocks to each GPU. Activations are passed between stages via point-to-point `dist.send`/`dist.recv`. This implementation uses synchronous (GPipe-style without micro-batching) execution, so one GPU is active at a time — creating a pipeline bubble.

**Implementation:** `src/parallelism/pipeline_parallel.py` — `PipelineParallelModel` splits the 12-layer GPT-small evenly (6 blocks/GPU at 2-GPU, 3 blocks/GPU at 4-GPU). Rank 0 holds embeddings, the last rank holds `ln_f` and `lm_head`.

```python
from src.parallelism.pipeline_parallel import PipelineParallelModel

# Split transformer blocks across GPUs
model = GPTModel(config).to(device)
model = PipelineParallelModel(model, world_size=world_size, rank=rank)
# rank 0: token_embed + pos_embed + blocks[0:3]   (4-GPU case)
# rank 1: blocks[3:6]
# rank 2: blocks[6:9]
# rank 3: blocks[9:12] + ln_f + lm_head  ← only rank with loss/backward

# Forward: activations flow rank-0 → rank-1 → … → rank-N via dist.send/recv
```

| GPUs | Precision | Throughput | Mem/GPU | vs 1-GPU same precision | Memory saved vs 1-GPU |
|------|-----------|-----------|---------|------------------------|-----------------------|
| 1 | fp32 | 8,300 tok/s | 5.21 GB | 1.00× (baseline) | — |
| 2 | fp32 | 5,481 tok/s | 1.92 GB | **0.66×** | **−63%** |
| 4 | fp32 | 6,411 tok/s | 1.30 GB | **0.77×** | **−75%** |
| 1 | fp16 | 26,097 tok/s | 4.56 GB | 1.00× (baseline) | — |
| 2 | fp16 | 16,733 tok/s | 1.67 GB | **0.64×** | **−63%** |
| 4 | fp16 | 19,333 tok/s | 1.18 GB | **0.74×** | **−74%** |

**Key insights:**
- **Memory is the headline:** each GPU holds only its assigned blocks + optimizer states for those blocks. At 4-GPU, peak memory drops from 5.21 GB → 1.30 GB per GPU — a **75% reduction**. This is what enables training models that don't fit on a single GPU.
- **4-GPU throughput (6,411) > 2-GPU (5,481) in fp32:** with fewer blocks per stage, each stage completes faster, reducing the per-step wall time despite 4× pipeline depth.
- **Throughput cost of the bubble:** without micro-batching, at any moment only one GPU is computing. This ~25–35% throughput overhead is why production systems (GPipe, PipeDream) use micro-batching to keep all GPUs active. That's the next implementation step.

---

## 🌐 Multi-Node FSDP

Extends single-node FSDP to clusters via `torchrun` distributed rendezvous.
NCCL handles inter-node AllReduce; FSDP shards parameters, gradients, and
optimizer states across all GPUs in the world (2 nodes × 4 GPUs = 8-way shard).

### Memory projection: Qwen2.5-7B on 2 nodes × 4 GPUs

| Component | 1 GPU (no FSDP) | 8 GPU FSDP (FULL_SHARD) |
|-----------|----------------|------------------------|
| Weights (bf16) | 15.2 GB | **1.9 GB/GPU** |
| Gradients | 15.2 GB | **1.9 GB/GPU** |
| Optimizer (AdamW fp32) | 30.4 GB | **3.8 GB/GPU** |
| Activations (seq=2048) | ~4 GB | ~4 GB/GPU |
| **Total** | **64.8 GB** | **~12 GB/GPU** → fits A30 24 GB |

Without multi-node FSDP, Qwen2.5-7B cannot be fine-tuned on a 24 GB GPU
(weights alone exceed memory). 8-way FSDP makes it feasible with room for
batch_size=2.

### SLURM launch (2 nodes)

```bash
sbatch scripts/launch/slurm_2node_fsdp.sh
# → srun torchrun --nnodes=2 --nproc_per_node=4 --node_rank=$SLURM_NODEID ...
```

### Manual launch (no SLURM)

```bash
# Node 0 (master, IP = 192.168.1.10):
MASTER_ADDR=192.168.1.10 NODE_RANK=0 NNODES=2 \
  bash scripts/launch/torchrun_multi_node.sh

# Node 1 (worker):
MASTER_ADDR=192.168.1.10 NODE_RANK=1 NNODES=2 \
  bash scripts/launch/torchrun_multi_node.sh
```

### Key changes vs single-node

| Setting | Single-node | Multi-node |
|---------|-------------|------------|
| torchrun | `--standalone` | `--nnodes=N --node_rank=K --master_addr=...` |
| Rendezvous | automatic | `--rdzv_backend=c10d --rdzv_endpoint=MASTER:PORT` |
| NCCL | intra-node NVLink/PCIe | inter-node Ethernet (`eth0`) or InfiniBand (`ib0`) |
| RANK env | 0…(gpus-1) | 0…(N×gpus-1) global; LOCAL_RANK resets per node |
| Timeout | default 10 min | `NCCL_TIMEOUT=1800` for large AllReduce |

Configs: [`configs/multi_node/`](configs/multi_node/) — 2-node 8-GPU, 4-node 16-GPU, Qwen2.5-7B FSDP.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────┐
│      Training Coordinator           │
│  Gradient Aggregation • Monitoring  │
└──────────┬──────────────────────────┘
           │
    ┌──────┴─────┬─────┬──────┐
    │            │     │      │
┌───▼──┐    ┌───▼──┐ ┌▼───┐ ┌▼────┐
│GPU 0 │    │GPU 1 │ │GPU 2│ │GPU 3│
│Model │    │Model │ │Model│ │Model│
└──┬───┘    └──┬───┘ └──┬──┘ └──┬──┘
   │           │        │       │
   └───────────┴────────┴───────┘
              │
    ┌─────────▼─────────┐
    │  NCCL All-Reduce  │
    │ (Gradient Sync)   │
    └───────────────────┘
```

**Core Components:**
1. **Data Parallelism** — PyTorch DDP & FSDP
2. **Communication** — NCCL backend (PCIe, 4× A30)
3. **Mixed Precision** — `torch.amp` autocast + GradScaler
4. **Fault Tolerance** — Checkpoint every N steps, auto-resume
5. **Parallelism** — Tensor and Pipeline parallelism also available

---

## 🔧 Implementation Deep-Dive

### FSDP — Drop-In for DDP (Option 1)

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy

# Before (DDP):
model = DDP(model, device_ids=[local_rank])

# After (FSDP — saves 12–21% memory per GPU):
model = FSDP(
    model,
    device_id=local_rank,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
)
```

At 4 GPUs: each A30 holds **1/4 of optimizer states** at rest, gathering and releasing parameter shards during the forward/backward pass via NCCL. Peak memory drops from 5.67 GB → 4.63 GB per GPU in fp32.

### Mixed Precision — `torch.amp` (Option 2)

```python
from torch.amp import autocast, GradScaler

scaler = GradScaler('cuda')

with autocast('cuda', dtype=torch.float16):
    loss = model(input_ids, labels=labels)['loss']   # fp16 forward

scaler.scale(loss).backward()   # scaled backward
scaler.step(optimizer)          # unscaled optimizer step (fp32 master weights)
scaler.update()
```

A30 Tensor Cores process fp16 matrix multiplications significantly faster than fp32. Result: **26,097 vs 8,300 tok/s** — 3.14× throughput with no model changes and no measurable accuracy degradation.

**Enable/disable in config:**
```yaml
training:
  mixed_precision_training: true   # fp16 AMP (recommended)
```

---

## 📈 Full Configuration Summary

All 18 configurations measured on 4× NVIDIA A30 (PCIe), 125M-param GPT, batch=4/GPU, seq=512.

| Strategy | GPUs | Precision | Throughput | Mem/GPU | Best For |
|----------|------|-----------|-----------|---------|----------|
| Single GPU | 1 | fp32 | 8,300 tok/s | 5.21 GB | fp32 reference |
| **Single GPU** | **1** | **fp16** | **26,097 tok/s** | **4.56 GB** | **Max single-GPU speed** |
| DDP | 2 | fp32 | 7,123 tok/s | 5.67 GB | — |
| DDP | 2 | fp16 | 20,522 tok/s | 5.02 GB | — |
| **FSDP** | **2** | **fp16** | **21,844 tok/s** | **4.31 GB** | **2-GPU, memory-aware** |
| DDP | 4 | fp32 | 10,921 tok/s | 5.67 GB | — |
| **DDP** | **4** | **fp16** | **20,950 tok/s** | **5.02 GB** | **4-GPU, max throughput** |
| FSDP | 4 | fp16 | 17,840 tok/s | 3.96 GB | 4-GPU, memory-constrained |
| TP | 2 | fp32 | 4,387 tok/s | 4.77 GB | Wide layers + NVLink hardware |
| TP | 4 | fp32 | 2,258 tok/s | 4.49 GB | Layers too wide for 1 GPU |
| **PP** | **2** | **fp32** | **5,481 tok/s** | **1.92 GB** | **Model too deep for 1 GPU** |
| **PP** | **4** | **fp32** | **6,411 tok/s** | **1.30 GB** | **−75% memory per GPU** |
| PP | 4 | fp16 | 19,333 tok/s | 1.18 GB | Max memory savings + AMP |

**Decision guide:**
- **Model fits in memory, maximize throughput** → DDP + AMP
- **Approaching memory limit** → FSDP + AMP (saves 12–21% with minimal throughput loss)
- **Model layers too large for one GPU** → Tensor Parallelism (requires NVLink for efficiency)
- **Model too deep for one GPU** → Pipeline Parallelism (75% memory reduction at 4 GPUs)
- **Single GPU** → AMP is the highest-ROI single change (3.14×)

---

## 🛠️ Technology Stack

- **Framework:** PyTorch 2.7.1+cu118
- **CUDA:** 11.8
- **Backend:** NCCL (PCIe interconnect)
- **Hardware:** 4× NVIDIA A30 (24 GB HBM2, PCIe)
- **Python:** 3.10

---

## 🎓 For Recruiters

### What This Project Demonstrates

**Technical Skills:**
- ✅ Distributed Systems Engineering — DDP and FSDP on multi-GPU hardware, with real measured efficiency
- ✅ Mixed Precision Training — 3.14× throughput from fp16 AMP; understand compute-memory tradeoffs
- ✅ Memory Optimization — FSDP: 12–21% reduction; Pipeline Parallel: 75% reduction at 4-GPU
- ✅ Tensor Parallelism — implemented gradient-aware `ParallelLinear` with custom `autograd.Function` for all_gather; measured 0.27× throughput at 4-GPU and diagnosed PCIe bottleneck vs NVLink requirement
- ✅ Pipeline Parallelism — split 12-layer GPT across 4 GPUs via send/recv; measured 75% memory reduction; identified pipeline bubble as root cause of throughput gap and next step (micro-batching)
- ✅ Performance Analysis — identified that PCIe bottleneck limits DDP efficiency; explained compute-to-communication ratio for TP vs PP tradeoffs
- ✅ Honest Benchmarking — reported actual numbers across 18 configurations; no synthetic projections

**Key Results (all measured on real hardware):**

> "Extended distributed training system to compare DDP vs FSDP, measuring per-GPU memory reduction from sharding optimizer states across 4 GPUs — FSDP cuts memory 18–21% per GPU at 4-GPU scale; identified throughput tradeoff (5% at fp32, 15% at fp16) and the regime where FSDP outperforms DDP on memory-constrained hardware."

> "Added automatic mixed precision (AMP) to the training loop using torch.amp; measured 3.14× throughput improvement (8,300 → 26,097 tok/s) on a 125M-param model on a single NVIDIA A30 — making AMP the highest-ROI optimization in the pipeline, ahead of adding a second GPU without AMP."

> "Implemented column-wise tensor parallelism from scratch using a custom gradient-aware autograd Function for all_gather; measured 0.27× throughput at 4-GPU on PCIe-connected A30s, diagnosing that 48 collective operations per training step saturates PCIe bandwidth — explains why TP requires NVLink at production scale."

> "Implemented synchronous pipeline parallelism splitting 12 transformer layers across 4 GPUs via point-to-point communication; measured 75% memory reduction per GPU (5.21 GB → 1.30 GB) and identified pipeline bubble as the key throughput cost — the next step is GPipe-style micro-batching to fill the bubble."

**Try It Yourself:**
👉 **[Interactive Demo - No Setup Required](https://distributed-training-models.streamlit.app/)**

---

## 🚀 Features

### Implemented & Benchmarked
- ✅ **Mixed precision training (fp16 AMP)** — 3.14× throughput (8,300 → 26,097 tok/s) on single A30
- ✅ **Multi-GPU DDP** — measured 1.32× total speedup at 4 GPUs (fp32, PCIe); explains PCIe bottleneck
- ✅ **Fully Sharded Data Parallel (FSDP)** — 12–21% memory reduction per GPU; 6.4% throughput gain at 2-GPU fp16
- ✅ **Tensor Parallelism** — column-wise weight sharding via `ParallelLinear`; gradient-aware all_gather; benchmarked at 2 and 4 GPU (communication-bound on PCIe, 0.27× at 4-GPU)
- ✅ **Pipeline Parallelism** — layer-wise block splitting via `PipelineParallelModel`; synchronous send/recv; benchmarked at 2 and 4 GPU (75% memory reduction at 4-GPU, 0.77× throughput)
- ✅ NCCL gradient synchronization
- ✅ Distributed data loading (`DistributedSampler`)
- ✅ Fault-tolerant checkpointing
- ✅ Gradient accumulation and gradient clipping
- ✅ Benchmark suite (`run_benchmark.py`, `run_tp_pp_benchmark.py`) — **18 configurations**, results in `benchmarks/real_results.json`

### Coming Soon
- 🔄 Multi-node training (Slurm script ready in `scripts/launch/`)
- 🔄 Flash Attention v2
- 🔄 ZeRO optimizer (DeepSpeed)

---

## 📚 Documentation

- **[Interactive Demo](https://distributed-training-models.streamlit.app/)** — Try it live
- **[Raw Benchmark Results](benchmarks/real_results.json)** — All measured numbers
- **[FSDP vs DDP Analysis](docs/FSDP_vs_DDP.md)** — In-depth comparison
- **[Tensor & Pipeline Parallelism](docs/TENSOR_PIPELINE_PARALLELISM.md)** — Advanced parallelism
- **[Results](RESULTS.md)** — Training run summaries

---

## 📄 License

MIT License — see [LICENSE](LICENSE)

---

## 🙏 Acknowledgments

Built with PyTorch, inspired by Megatron-LM and nanoGPT.

---

## 📬 Contact

**Sai Teja Srivilli**  
📧 srivillibhutturu.s@northeastern.edu  
🐙 [GitHub](https://github.com/saitejasrivilli)  
🎮 [Live Demo](https://distributed-training-models.streamlit.app/)

---

<p align="center">
  <strong>⭐ Star this repo if you found it helpful!</strong>
</p>

<p align="center">
  <sub>Built with ❤️ for the ML community</sub>
</p>
