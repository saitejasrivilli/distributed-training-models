#!/usr/bin/env python3
"""Main training script for distributed LLM training"""

import sys
import os

# Force unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

print("="*60, flush=True)
print("🚀 Starting training script...", flush=True)
print("="*60, flush=True)

try:
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    import argparse
    import yaml
    from pathlib import Path
    
    print("✅ Imports successful", flush=True)
    
    from src.model.transformer import GPTModel
    from src.model.config import CONFIGS
    from src.data.dataloader import get_dataloader
    from src.training.trainer import Trainer
    from src.utils.device import setup_distributed, cleanup_distributed
    from src.utils.checkpoint import save_checkpoint
    
    print("✅ Local imports successful", flush=True)

except Exception as e:
    print(f"❌ Import error: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)

def load_config(config_path):
    """Load YAML configuration"""
    print(f"Loading config from: {config_path}", flush=True)
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    print("🏁 Entering main function...", flush=True)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train LLM from scratch')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--output_dir', type=str, default='experiments/run', help='Output directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    args = parser.parse_args()
    
    print(f"Arguments parsed: {args}", flush=True)
    
    # Load config
    config = load_config(args.config)
    config['training']['output_dir'] = args.output_dir
    
    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    is_main = (rank == 0)
    
    if is_main:
        print("=" * 60)
        print("🚀 Distributed LLM Training")
        print("=" * 60)
        print(f"World Size: {world_size}")
        print(f"Config: {args.config}")
        print(f"Output: {args.output_dir}")
        print("=" * 60)
    
    # Set device
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model_config = CONFIGS[config['model']['name']]
    model = GPTModel(model_config)
    model = model.to(device)
    
    if is_main:
        print(f"\n📊 Model: {config['model']['name']}")
        print(f"Parameters: {model_config.n_params:,}")
    
    # Wrap with DDP
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
    
    # Create dataloader
    if is_main:
        print(f"\n📁 Loading dataset: {config['data']['dataset_name']}")
    
    dataloader, tokenizer = get_dataloader(config, rank, world_size)
    
    if is_main:
        print(f"Dataset loaded: {len(dataloader)} batches")
    
    # Create trainer
    trainer = Trainer(model, config, rank, world_size)
    
    # Training
    if is_main:
        print(f"\n🏋️  Starting training for {config['training']['max_steps']} steps...")
    
    try:
        trainer.train(dataloader, config['training']['max_steps'])
        
        if is_main:
            print("\n✅ Training completed!")
            
            # Save final checkpoint
            save_checkpoint(
                model.module if world_size > 1 else model,
                trainer.optimizer,
                trainer.step,
                trainer.current_loss,
                args.output_dir,
                rank
            )
            
            # Create training summary
            import json
            summary = {
                'config': args.config,
                'final_step': trainer.step,
                'final_loss': float(trainer.current_loss),
                'num_gpus': world_size,
                'model_params': model_config.n_params,
            }
            
            summary_path = Path(args.output_dir) / 'training_summary.json'
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"📊 Training summary saved to: {summary_path}")
    
    except KeyboardInterrupt:
        if is_main:
            print("\n⚠️  Training interrupted by user")
            save_checkpoint(
                model.module if world_size > 1 else model,
                trainer.optimizer,
                trainer.step,
                trainer.current_loss,
                args.output_dir,
                rank
            )
    
    except Exception as e:
        if is_main:
            print(f"\n❌ Error during training: {e}")
            import traceback
            traceback.print_exc()
        raise
    
    finally:
        cleanup_distributed()
        if is_main:
            print("\n👋 Done!")

if __name__ == "__main__":
    print("Script starting...", flush=True)
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
