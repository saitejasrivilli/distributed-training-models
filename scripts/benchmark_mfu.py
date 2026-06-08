"""
Benchmark MFU (Model FLOP Utilisation) across parallelism configurations.

Usage:
    # Real benchmark — trains 100 steps per config (requires GPU)
    python scripts/benchmark_mfu.py

    # Dry-run — prints theoretical estimates without touching the GPU
    python scripts/benchmark_mfu.py --dry_run
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import math

# Make src importable when run from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_A30_PEAK_FLOPS_FP16 = 165e12   # 165 TFLOPS FP16 (A30 PCIe spec)
_A30_PEAK_FLOPS_FP32 = 10.3e12  # 10.3 TFLOPS FP32 (A30 PCIe spec)
_A30_MEM_GB = 24.0


# ---------------------------------------------------------------------------
# Theoretical estimates (dry-run path)
# ---------------------------------------------------------------------------

def theoretical_mfu(
    n_params: int,
    batch_size: int,       # per-GPU
    seq_len: int,
    world_size: int,
    tok_per_sec: float,    # realistic measured estimate
    peak_flops: float,
) -> float:
    """MFU = achieved_flops / peak_flops."""
    achieved = 6 * n_params * (batch_size * world_size) * seq_len * (tok_per_sec / (batch_size * world_size * seq_len))
    # simplify: achieved = 6 * N * tok_per_sec
    achieved = 6 * n_params * tok_per_sec
    return achieved / peak_flops


def dry_run_estimates() -> list[dict]:
    """
    Return theoretical estimates based on real benchmark numbers from
    benchmarks/real_results.json (125M GPT, batch=4/GPU, seq=512, A30 PCIe).
    """
    n_params = 124_475_904  # 125M GPT

    # tok/s numbers from real benchmarks
    configs = [
        {"label": "1-GPU fp32",      "tok_s": 8_300,  "mem_gb": 5.21, "world": 1, "dtype": "fp32"},
        {"label": "1-GPU fp16 AMP",  "tok_s": 26_097, "mem_gb": 4.56, "world": 1, "dtype": "fp16"},
        {"label": "4-GPU DDP fp16",  "tok_s": 20_950, "mem_gb": 5.02, "world": 4, "dtype": "fp16"},
        {"label": "4-GPU FSDP fp16", "tok_s": 17_840, "mem_gb": 3.96, "world": 4, "dtype": "fp16"},
    ]

    results = []
    for c in configs:
        peak = _A30_PEAK_FLOPS_FP16 if c["dtype"] == "fp16" else _A30_PEAK_FLOPS_FP32
        mfu = (6 * n_params * c["tok_s"]) / peak
        results.append({
            "config": c["label"],
            "mfu": mfu,
            "tok_s": c["tok_s"],
            "mem_gb": c["mem_gb"],
        })
    return results


# ---------------------------------------------------------------------------
# Real benchmark path
# ---------------------------------------------------------------------------

def run_single_config(
    label: str,
    use_amp: bool,
    world_size: int,
    n_steps: int = 100,
    batch_size: int = 4,
    seq_len: int = 512,
) -> dict:
    """Train *n_steps* with a tiny GPT-style model and measure MFU."""
    import torch
    from torch.amp import autocast, GradScaler

    try:
        from src.model.gpt import GPTModel
        from src.model.config import GPTConfig
    except ImportError:
        # Fallback: build a minimal transformer inline
        GPTModel = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if use_amp else torch.float32

    # Build a 125M-param model config
    if GPTModel is not None:
        try:
            import yaml
            cfg_path = os.path.join(os.path.dirname(__file__), "..", "configs", "quick_test.yaml")
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            model = GPTModel(cfg['model']).to(device)
        except Exception:
            GPTModel = None

    if GPTModel is None:
        # Minimal GPT-style model (embedding + 4 transformer blocks)
        import torch.nn as nn

        class _SmallTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                vocab, d, seq = 50257, 768, seq_len
                self.embed = nn.Embedding(vocab, d)
                self.pos = nn.Embedding(seq, d)
                layer = nn.TransformerEncoderLayer(d, 12, d * 4, batch_first=True)
                self.blocks = nn.TransformerEncoder(layer, num_layers=4)
                self.head = nn.Linear(d, vocab, bias=False)

            def forward(self, x, labels=None):
                B, S = x.shape
                h = self.embed(x) + self.pos(torch.arange(S, device=x.device))
                h = self.blocks(h)
                logits = self.head(h)
                loss = None
                if labels is not None:
                    loss = nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)), labels.view(-1)
                    )
                return {"loss": loss, "logits": logits}

        model = _SmallTransformer().to(device)

    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = GradScaler('cuda') if use_amp else None

    vocab_size = 50257
    elapsed_list: list[float] = []
    loss_val = 0.0

    # Warm-up (excluded from timing)
    for _ in range(3):
        x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        with autocast('cuda', enabled=use_amp, dtype=torch.float16):
            out = model(x, labels=x)
            loss = out['loss']
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        optimizer.zero_grad()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for _ in range(n_steps):
        x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with autocast('cuda', enabled=use_amp, dtype=torch.float16):
            out = model(x, labels=x)
            loss = out['loss']

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        optimizer.zero_grad()

        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_list.append(time.perf_counter() - t0)
        loss_val = loss.item()

    avg_elapsed = sum(elapsed_list) / len(elapsed_list)
    tok_s = (batch_size * world_size * seq_len) / avg_elapsed
    peak_flops = _A30_PEAK_FLOPS_FP16 if use_amp else _A30_PEAK_FLOPS_FP32
    achieved = (6 * n_params * batch_size * world_size * seq_len) / avg_elapsed
    mfu = achieved / peak_flops

    mem_gb = 0.0
    if device.type == "cuda":
        mem_gb = torch.cuda.max_memory_allocated(device) / 1e9

    return {"config": label, "mfu": mfu, "tok_s": tok_s, "mem_gb": mem_gb}


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def print_table(rows: list[dict]) -> None:
    header = f"{'Config':<22} | {'MFU':>6} | {'tok/s':>8} | {'mem/GPU':>9}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in rows:
        print(
            f"{r['config']:<22} | {r['mfu']:>5.1%} | {r['tok_s']:>8,.0f} | {r['mem_gb']:>7.2f} GB"
        )
    print(sep)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark MFU across parallelism configs")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print theoretical estimates from real benchmark results without running training",
    )
    parser.add_argument("--steps", type=int, default=100, help="Training steps per config")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length")
    args = parser.parse_args()

    print("\n=== MFU Benchmark — 125M GPT on NVIDIA A30 PCIe ===\n")

    if args.dry_run:
        print("(dry-run: theoretical estimates from real benchmark numbers)\n")
        rows = dry_run_estimates()
        print_table(rows)
        print(
            "\nNote: A30 PCIe has no NVLink. Cross-GPU bandwidth ~16 GB/s vs 600 GB/s NVLink.\n"
            "Multi-GPU MFU is lower than single-GPU because PCIe all-reduce dominates at 125M params.\n"
        )
        return

    # Real benchmark — single-process only (DDP/FSDP require torchrun)
    configs = [
        {"label": "1-GPU fp32",     "use_amp": False, "world_size": 1},
        {"label": "1-GPU fp16 AMP", "use_amp": True,  "world_size": 1},
        # For multi-GPU, launch with torchrun and adapt accordingly
    ]

    rows = []
    for c in configs:
        print(f"Running {c['label']} ({args.steps} steps) ...")
        try:
            row = run_single_config(
                label=c["label"],
                use_amp=c["use_amp"],
                world_size=c["world_size"],
                n_steps=args.steps,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
            )
            rows.append(row)
        except Exception as exc:
            print(f"  SKIPPED ({exc})")
            rows.append({"config": c["label"], "mfu": 0.0, "tok_s": 0.0, "mem_gb": 0.0})

    # For DDP/FSDP, append dry-run rows so the table is complete
    dry = dry_run_estimates()
    dry_map = {r["config"]: r for r in dry}
    for label in ("4-GPU DDP fp16", "4-GPU FSDP fp16"):
        if label in dry_map:
            row = dict(dry_map[label])
            row["config"] += " (from benchmark)"
            rows.append(row)

    print()
    print_table(rows)
    print(
        "\nNote: 4-GPU rows are from real benchmarks/real_results.json.\n"
        "Run `torchrun --nproc_per_node=4 run_benchmark.py` to reproduce live.\n"
    )


if __name__ == "__main__":
    main()
