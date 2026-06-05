#!/usr/bin/env python3
"""Benchmark script comparing DDP, FSDP, Tensor, Pipeline parallelism"""

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.amp import autocast, GradScaler
import time
import json
from pathlib import Path
import argparse
import yaml

from src.model.transformer import GPTModel
from src.model.config import CONFIGS
from src.data.dataloader import get_dataloader
from src.utils.device import setup_distributed, cleanup_distributed
from src.parallelism import convert_linear_to_tensor_parallel, split_model_pipeline


def benchmark_config(model, config, dataloader, strategy='ddp', use_amp=False, num_steps=100, rank=0, world_size=1):
    """Benchmark a specific configuration"""
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

    # Setup model
    model = model.to(device)

    # Wrap model
    if world_size > 1:
        if strategy == 'fsdp':
            model = FSDP(
                model,
                device_id=device,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                auto_wrap_policy=None,
            )
        elif strategy == 'tensor':
            model = convert_linear_to_tensor_parallel(model, world_size=world_size, rank=rank)
            model = DDP(model, device_ids=[rank])
        elif strategy == 'pipeline':
            model = split_model_pipeline(model, world_size=world_size, rank=rank)
        else:
            model = DDP(model, device_ids=[rank])

    # Setup optimizer and training
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        betas=(config['training']['beta1'], config['training']['beta2']),
        weight_decay=config['training']['weight_decay']
    )

    scaler = GradScaler('cuda') if use_amp else None
    model.train()

    # Warmup
    for i, batch in enumerate(dataloader):
        if i >= 5:
            break
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        with autocast('cuda', enabled=use_amp, dtype=torch.float16):
            outputs = model(input_ids, labels=labels)
            loss = outputs['loss']

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['grad_clip'])

        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        optimizer.zero_grad()

    if world_size > 1:
        dist.barrier()

    # Benchmark
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()

    for i, batch in enumerate(dataloader):
        if i >= num_steps:
            break

        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        with autocast('cuda', enabled=use_amp, dtype=torch.float16):
            outputs = model(input_ids, labels=labels)
            loss = outputs['loss']

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['grad_clip'])

        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        optimizer.zero_grad()

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.time() - start_time

    # Memory stats
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
        reserved = torch.cuda.max_memory_allocated(device) / 1024**3  # GB
    else:
        allocated = reserved = 0

    throughput = (num_steps * config['training']['batch_size']) / elapsed

    return {
        'elapsed_time': elapsed,
        'throughput_samples_per_sec': throughput,
        'allocated_memory_gb': allocated,
        'reserved_memory_gb': reserved,
        'strategy': strategy,
        'use_amp': use_amp,
        'num_steps': num_steps,
    }


def main():
    parser = argparse.ArgumentParser(description='Benchmark training configurations')
    parser.add_argument('--config', type=str, default='configs/data_parallel/train_117M.yaml', help='Config file')
    parser.add_argument('--num_steps', type=int, default=100, help='Number of steps to benchmark')
    parser.add_argument('--output_dir', type=str, default='benchmarks', help='Output directory')
    args = parser.parse_args()

    # Setup
    rank, world_size, local_rank = setup_distributed()
    is_main = (rank == 0)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Load dataloader
    dataloader, _ = get_dataloader(config, rank, world_size)

    # Run benchmarks
    results = {}

    if is_main:
        print("\n" + "="*60)
        print("🔬 Training Configuration Benchmarks")
        print("="*60)

    configs_to_test = [
        ('ddp', False, 'DDP without mixed precision'),
        ('ddp', True, 'DDP with mixed precision (fp16)'),
        ('fsdp', False, 'FSDP without mixed precision'),
        ('fsdp', True, 'FSDP with mixed precision (fp16)'),
        ('tensor', False, 'Tensor Parallelism without mixed precision'),
        ('tensor', True, 'Tensor Parallelism with mixed precision (fp16)'),
        ('pipeline', False, 'Pipeline Parallelism without mixed precision'),
        ('pipeline', True, 'Pipeline Parallelism with mixed precision (fp16)'),
    ]

    for strategy, use_amp, desc in configs_to_test:
        if is_main:
            print(f"\nBenchmarking: {desc}")

        model_config = CONFIGS[config['model']['name']]
        model = GPTModel(model_config)

        result = benchmark_config(
            model,
            config,
            dataloader,
            strategy=strategy,
            use_amp=use_amp,
            num_steps=args.num_steps,
            rank=rank,
            world_size=world_size
        )

        key = f"{strategy}_amp{int(use_amp)}"
        results[key] = result

        if is_main:
            print(f"  ✓ Time: {result['elapsed_time']:.2f}s")
            print(f"  ✓ Throughput: {result['throughput_samples_per_sec']:.0f} samples/sec")
            print(f"  ✓ Memory allocated: {result['allocated_memory_gb']:.2f} GB")

    if is_main:
        # Compute comparisons
        ddp_baseline = results.get('ddp_amp0', {}).get('throughput_samples_per_sec', 0)
        if ddp_baseline > 0:
            for key in results:
                throughput = results[key]['throughput_samples_per_sec']
                results[key]['relative_throughput'] = throughput / ddp_baseline

        # Save results
        Path(args.output_dir).mkdir(exist_ok=True)
        output_file = Path(args.output_dir) / 'benchmark_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print("\n" + "="*60)
        print("📊 Benchmark Results Summary")
        print("="*60)
        for key, result in results.items():
            print(f"\n{key}:")
            print(f"  Throughput: {result['throughput_samples_per_sec']:.0f} samples/sec")
            if 'relative_throughput' in result:
                print(f"  Relative speedup: {result['relative_throughput']:.2f}x")
            print(f"  Memory: {result['allocated_memory_gb']:.2f} GB")
            print(f"  Time: {result['elapsed_time']:.2f}s")

        print(f"\n✅ Results saved to {output_file}")

    cleanup_distributed()


if __name__ == '__main__':
    main()
