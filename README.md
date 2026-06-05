# 🚀 Distributed LLM Pre-Training from Scratch

[![Live Demo](https://img.shields.io/badge/Demo-Live%20on%20Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://distributed-training-models.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.7](https://img.shields.io/badge/pytorch-2.7.1-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![AMP Speedup](https://img.shields.io/badge/AMP%20Speedup-3.14x%20single%20GPU-orange)](https://github.com/saitejasrivilli/distributed-training-models)
[![FSDP Memory](https://img.shields.io/badge/FSDP-up%20to%2021%25%20memory%20saved-brightgreen)](https://github.com/saitejasrivilli/distributed-training-models)

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
- **10 configurations** benchmarked
- 1 / 2 / 4 GPU × DDP, FSDP, fp32/fp16
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

# Reproduce the full benchmark (1 GPU then 2/4 GPU)
python run_benchmark.py
torchrun --nproc_per_node=2 run_benchmark.py
torchrun --nproc_per_node=4 run_benchmark.py
# → writes benchmarks/real_results.json
```

---

## 📊 Real Benchmark Results

> Measured on **NVIDIA A30 GPUs (PCIe)**, **163M-param GPT model**, batch=4/GPU, seq_len=512.  
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

**Key insight:** AMP delivers **3.14× throughput** on a single A30 GPU — more than doubling the benefit of adding a second GPU at this model scale. For a 163M model on PCIe-connected GPUs, AMP is the highest-ROI optimization.

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

Honest numbers for PCIe-connected A30 GPUs at 163M params:

| GPUs | Strategy | Precision | Total Throughput | Speedup vs 1-GPU fp32 | Parallel Efficiency |
|------|----------|-----------|-----------------|----------------------|---------------------|
| 1 | — | fp32 | 8,300 tok/s | 1.00× | 100% |
| 1 | — | **fp16** | **26,097 tok/s** | **3.14×** | — (AMP, not scaling) |
| 2 | DDP | fp32 | 7,123 tok/s | 0.86× | 42.9% |
| 2 | DDP | fp16 | 20,522 tok/s | 2.47× | — |
| 4 | DDP | fp32 | 10,921 tok/s | 1.32× | 32.9% |
| 4 | DDP | fp16 | 20,950 tok/s | 2.52× | — |

**Why efficiency is low:** A30 GPUs here are PCIe-connected (no NVLink). Each DDP step all-reduces ~650 MB of fp32 gradients across GPUs. At 163M params, the compute-to-communication ratio is unfavorable — DDP efficiency typically improves significantly with larger models (>1B params) or NVLink interconnects, where compute dominates communication.

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

| Strategy | GPUs | Precision | Throughput | Mem/GPU | Best For |
|----------|------|-----------|-----------|---------|----------|
| Single GPU | 1 | fp32 | 8,300 tok/s | 5.21 GB | fp32 reference |
| **Single GPU** | 1 | **fp16** | **26,097 tok/s** | **4.56 GB** | **Max single-GPU speed** |
| DDP | 2 | fp32 | 7,123 tok/s | 5.67 GB | — |
| DDP | 2 | fp16 | 20,522 tok/s | 5.02 GB | — |
| **FSDP** | 2 | **fp16** | **21,844 tok/s** | **4.31 GB** | **2-GPU, memory-aware** |
| DDP | 4 | fp32 | 10,921 tok/s | 5.67 GB | — |
| **DDP** | 4 | **fp16** | **20,950 tok/s** | **5.02 GB** | **4-GPU, max throughput** |
| FSDP | 4 | fp16 | 17,840 tok/s | 3.96 GB | 4-GPU, memory-constrained |

**Decision guide:**
- **Model fits in GPU memory, maximize speed** → DDP + AMP
- **Approaching memory limit** → FSDP + AMP
- **Single GPU available** → AMP is the most impactful single change

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
- ✅ Memory Optimization — FSDP reduces per-GPU memory 12–21% by sharding optimizer states; savings grow with GPU count
- ✅ Performance Analysis — identified that PCIe bottleneck limits DDP efficiency at 163M params; explained the compute-to-communication ratio
- ✅ Production ML Infrastructure — checkpointing, distributed data loading, gradient clipping, gradient accumulation
- ✅ Honest Benchmarking — reported actual scaling efficiency (32–43%) not marketing numbers

**Key Results (all measured on real hardware):**

> "Extended distributed training system to compare DDP vs FSDP, measuring per-GPU memory reduction from sharding optimizer states across 4 GPUs — FSDP cuts memory 18–21% per GPU at 4-GPU scale; identified throughput tradeoff (5% at fp32, 15% at fp16) and the regime where FSDP outperforms DDP on memory-constrained hardware."

> "Added automatic mixed precision (AMP) to the training loop using torch.amp; measured 3.14× throughput improvement (8,300 → 26,097 tok/s) on a 163M-param model on a single NVIDIA A30 — making AMP the highest-ROI optimization in the pipeline, ahead of adding a second GPU without AMP."

**Try It Yourself:**
👉 **[Interactive Demo - No Setup Required](https://distributed-training-models.streamlit.app/)**

---

## 🚀 Features

### Implemented & Benchmarked
- ✅ Multi-GPU data parallelism (DDP) — measured 1.32× total speedup at 4 GPUs (fp32, PCIe)
- ✅ **Fully Sharded Data Parallel (FSDP)** — 12–21% memory reduction per GPU measured
- ✅ **Mixed precision training (fp16 AMP)** — 3.14× throughput on single A30
- ✅ Tensor Parallelism — column-wise linear layer sharding
- ✅ Pipeline Parallelism — layer-wise model sharding
- ✅ NCCL gradient synchronization
- ✅ Distributed data loading (`DistributedSampler`)
- ✅ Fault-tolerant checkpointing
- ✅ Gradient accumulation and gradient clipping
- ✅ Benchmark suite (`run_benchmark.py`) — 10 configurations, results in `benchmarks/real_results.json`

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
