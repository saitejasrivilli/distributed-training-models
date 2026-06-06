#!/usr/bin/env python3
"""
Real benchmark: single-GPU, DDP, FSDP × fp32/fp16.
Run single-GPU:  python run_benchmark.py
Run multi-GPU:   torchrun --nproc_per_node=4 run_benchmark.py
Results saved to benchmarks/real_results.json
"""

import os, time, json, argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.amp import autocast, GradScaler
from pathlib import Path

from src.model.transformer import GPTModel
from src.model.config import CONFIGS
from src.utils.device import setup_distributed, cleanup_distributed


BATCH_SIZE = 4        # per GPU
SEQ_LEN    = 512
NUM_WARMUP = 5
NUM_STEPS  = 50


def make_batch(batch_size, seq_len, device):
    ids = torch.randint(0, 50257, (batch_size, seq_len), device=device)
    return ids, ids.clone()


def run_one(model, device, use_amp, num_warmup, num_steps):
    """Return (tokens_per_sec, peak_mem_gb)."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = GradScaler('cuda') if use_amp else None
    torch.cuda.reset_peak_memory_stats(device)
    model.train()

    for _ in range(num_warmup):
        ids, labels = make_batch(BATCH_SIZE, SEQ_LEN, device)
        with autocast('cuda', enabled=use_amp, dtype=torch.float16):
            loss = model(ids, labels=labels)['loss']
        if use_amp:
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            loss.backward(); optimizer.step()
        optimizer.zero_grad()

    torch.cuda.synchronize(device)
    t0 = time.time()

    for _ in range(num_steps):
        ids, labels = make_batch(BATCH_SIZE, SEQ_LEN, device)
        with autocast('cuda', enabled=use_amp, dtype=torch.float16):
            loss = model(ids, labels=labels)['loss']
        if use_amp:
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            loss.backward(); optimizer.step()
        optimizer.zero_grad()

    torch.cuda.synchronize(device)
    elapsed  = time.time() - t0
    tok_s    = (num_steps * BATCH_SIZE * SEQ_LEN) / elapsed
    peak_gb  = torch.cuda.max_memory_allocated(device) / 1024**3
    return tok_s, peak_gb


def main():
    rank, world_size, local_rank = setup_distributed()
    is_main = (rank == 0)
    device  = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    if is_main:
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
        print(f"\n{'='*60}")
        print(f"Benchmark  |  world_size={world_size}  |  {gpu_name}")
        print(f"Batch/GPU={BATCH_SIZE}  seq_len={SEQ_LEN}  steps={NUM_STEPS}")
        print(f"{'='*60}")

    model_cfg = CONFIGS['small']   # 125M-param model
    results   = {}

    configs = []
    if world_size == 1:
        configs = [
            ('single_fp32', 'none', False),
            ('single_fp16', 'none', True),
        ]
    else:
        configs = [
            ('ddp_fp32',  'ddp',  False),
            ('ddp_fp16',  'ddp',  True),
            ('fsdp_fp32', 'fsdp', False),
            ('fsdp_fp16', 'fsdp', True),
        ]

    for key_base, strategy, use_amp in configs:
        key = f"gpu{world_size}_{key_base}"
        # fresh model every run
        model = GPTModel(model_cfg, use_gradient_checkpointing=False).to(device)

        if strategy == 'ddp':
            model = DDP(model, device_ids=[local_rank])
        elif strategy == 'fsdp':
            model = FSDP(
                model,
                device_id=local_rank,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                auto_wrap_policy=None,
            )

        if world_size > 1:
            dist.barrier()

        label = f"{'AMP fp16' if use_amp else 'fp32':8s} | {strategy.upper() if strategy != 'none' else 'single-GPU'}"
        if is_main:
            print(f"\nRunning: {label} ...")

        tok_s, mem_gb = run_one(model, device, use_amp, NUM_WARMUP, NUM_STEPS)

        # Aggregate throughput across ranks
        if world_size > 1:
            t = torch.tensor(tok_s, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            tok_s = t.item()

        if is_main:
            results[key] = {
                'strategy':       strategy,
                'use_amp':        use_amp,
                'world_size':     world_size,
                'tokens_per_sec': round(tok_s),
                'peak_mem_gb':    round(mem_gb, 2),
                'batch_per_gpu':  BATCH_SIZE,
                'seq_len':        SEQ_LEN,
                'model_params':   model_cfg.n_params,
            }
            print(f"  tokens/sec : {tok_s:,.0f}")
            print(f"  peak mem   : {mem_gb:.2f} GB/GPU")

        del model
        torch.cuda.empty_cache()

    if is_main:
        Path('benchmarks').mkdir(exist_ok=True)
        out = Path('benchmarks/real_results.json')
        # merge with any previous run
        existing = json.loads(out.read_text()) if out.exists() else {}
        existing.update(results)
        out.write_text(json.dumps(existing, indent=2))
        print(f"\n{'='*60}")
        print(f"Results saved -> {out}")

    cleanup_distributed()


if __name__ == '__main__':
    main()
