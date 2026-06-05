#!/usr/bin/env python3
"""
Tensor Parallel and Pipeline Parallel benchmark.
Run 2-GPU:  torchrun --nproc_per_node=2 --master_port=29510 run_tp_pp_benchmark.py
Run 4-GPU:  torchrun --nproc_per_node=4 --master_port=29511 run_tp_pp_benchmark.py
Results merged into benchmarks/real_results.json
"""

import os, time, json
import torch
import torch.distributed as dist
from torch.amp import autocast, GradScaler
from pathlib import Path

from src.model.transformer import GPTModel
from src.model.config import CONFIGS
from src.utils.device import setup_distributed, cleanup_distributed
from src.parallelism.tensor_parallel import convert_linear_to_tensor_parallel
from src.parallelism.pipeline_parallel import PipelineParallelModel

BATCH_SIZE = 4
SEQ_LEN    = 512
NUM_WARMUP = 3
NUM_STEPS  = 30


def make_batch(device):
    ids = torch.randint(0, 50257, (BATCH_SIZE, SEQ_LEN), device=device)
    return ids, ids.clone()


# ---------------------------------------------------------------------------
# Tensor Parallel benchmark
# ---------------------------------------------------------------------------

def bench_tp(model_cfg, device, world_size, rank, use_amp):
    model = GPTModel(model_cfg, use_gradient_checkpointing=False).to(device)

    # Skip lm_head: vocab_size 50257 is not divisible by 2 or 4
    model = convert_linear_to_tensor_parallel(
        model, world_size=world_size, rank=rank, skip_modules={'lm_head'}
    )
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = GradScaler('cuda') if use_amp else None
    torch.cuda.reset_peak_memory_stats(device)

    for _ in range(NUM_WARMUP):
        ids, labels = make_batch(device)
        with autocast('cuda', enabled=use_amp, dtype=torch.float16):
            loss = model(ids, labels=labels)['loss']
        if use_amp:
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            loss.backward(); optimizer.step()
        optimizer.zero_grad()
        dist.barrier()

    torch.cuda.synchronize(device)
    dist.barrier()
    t0 = time.time()

    for _ in range(NUM_STEPS):
        ids, labels = make_batch(device)
        with autocast('cuda', enabled=use_amp, dtype=torch.float16):
            loss = model(ids, labels=labels)['loss']
        if use_amp:
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            loss.backward(); optimizer.step()
        optimizer.zero_grad()
        dist.barrier()

    torch.cuda.synchronize(device)
    elapsed  = time.time() - t0
    # All ranks do the same forward; throughput = per-rank tokens (same batch replicated)
    tok_s    = (NUM_STEPS * BATCH_SIZE * SEQ_LEN) / elapsed
    peak_gb  = torch.cuda.max_memory_allocated(device) / 1024**3

    del model
    torch.cuda.empty_cache()
    return tok_s, peak_gb


# ---------------------------------------------------------------------------
# Pipeline Parallel benchmark
# ---------------------------------------------------------------------------

def bench_pp(model_cfg, device, world_size, rank, use_amp):
    base_model = GPTModel(model_cfg, use_gradient_checkpointing=False).to(device)
    model = PipelineParallelModel(base_model, world_size=world_size, rank=rank)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = GradScaler('cuda') if use_amp else None
    torch.cuda.reset_peak_memory_stats(device)
    is_last = (rank == world_size - 1)

    for _ in range(NUM_WARMUP):
        ids, labels = make_batch(device)
        with autocast('cuda', enabled=use_amp, dtype=torch.float16):
            out = model(ids, labels=labels)
        if is_last and out['loss'] is not None:
            loss = out['loss']
            if use_amp:
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            else:
                loss.backward(); optimizer.step()
        optimizer.zero_grad()
        dist.barrier()

    torch.cuda.synchronize(device)
    dist.barrier()
    t0 = time.time()

    for _ in range(NUM_STEPS):
        ids, labels = make_batch(device)
        with autocast('cuda', enabled=use_amp, dtype=torch.float16):
            out = model(ids, labels=labels)
        if is_last and out['loss'] is not None:
            loss = out['loss']
            if use_amp:
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            else:
                loss.backward(); optimizer.step()
        optimizer.zero_grad()
        dist.barrier()

    torch.cuda.synchronize(device)
    elapsed  = time.time() - t0
    # Pipeline throughput: tokens flowing through the full pipeline per second
    tok_s    = (NUM_STEPS * BATCH_SIZE * SEQ_LEN) / elapsed
    peak_gb  = torch.cuda.max_memory_allocated(device) / 1024**3

    del model
    torch.cuda.empty_cache()
    return tok_s, peak_gb


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rank, world_size, local_rank = setup_distributed()
    is_main = (rank == 0)
    device  = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    model_cfg = CONFIGS['small']

    if is_main:
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
        print(f"\n{'='*60}")
        print(f"TP/PP Benchmark  |  world_size={world_size}  |  {gpu}")
        print(f"Batch/GPU={BATCH_SIZE}  seq={SEQ_LEN}  warmup={NUM_WARMUP}  steps={NUM_STEPS}")
        print(f"{'='*60}")

    results = {}

    configs = [
        ('tp_fp32',  'tp',  False),
        ('tp_fp16',  'tp',  True),
        ('pp_fp32',  'pp',  False),
        ('pp_fp16',  'pp',  True),
    ]

    for key_base, strategy, use_amp in configs:
        key   = f"gpu{world_size}_{key_base}"
        label = f"{'AMP fp16' if use_amp else 'fp32':8s} | {strategy.upper()}"
        if is_main:
            print(f"\nRunning: {label} ...")

        try:
            if strategy == 'tp':
                tok_s, mem_gb = bench_tp(model_cfg, device, world_size, rank, use_amp)
            else:
                tok_s, mem_gb = bench_pp(model_cfg, device, world_size, rank, use_amp)

            if is_main:
                results[key] = {
                    'strategy':       strategy,
                    'use_amp':        use_amp,
                    'world_size':     world_size,
                    'tokens_per_sec': round(tok_s),
                    'peak_mem_gb':    round(mem_gb, 2),
                    'batch_per_gpu':  BATCH_SIZE,
                    'seq_len':        SEQ_LEN,
                }
                print(f"  tokens/sec : {tok_s:,.0f}")
                print(f"  peak mem   : {mem_gb:.2f} GB/GPU")

        except Exception as e:
            if is_main:
                print(f"  ERROR: {e}")
            dist.barrier()

    if is_main and results:
        Path('benchmarks').mkdir(exist_ok=True)
        out = Path('benchmarks/real_results.json')
        existing = json.loads(out.read_text()) if out.exists() else {}
        existing.update(results)
        out.write_text(json.dumps(existing, indent=2))
        print(f"\n{'='*60}")
        print(f"Results saved -> {out}")

    cleanup_distributed()


if __name__ == '__main__':
    main()
