import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import CPUOffload, ShardingStrategy
import torch.distributed as dist
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import os

class Trainer:
    """Main training class"""
    
    def __init__(self, model, config, rank=0, world_size=1):
        self.model = model
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.is_main = (rank == 0)
        self.device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

        # Mixed precision setup
        self.use_mixed_precision = config['training'].get('mixed_precision_training', True)
        self.scaler = GradScaler('cuda') if self.use_mixed_precision else None

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            betas=(config['training']['beta1'], config['training']['beta2']),
            weight_decay=config['training']['weight_decay']
        )

        # Training state
        self.step = 0
        self.epoch = 0
        self.current_loss = 0.0
        
    def train_step(self, batch, is_accumulating=False):
        """Single training step. When is_accumulating=True, skips optimizer step."""
        self.model.train()

        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)

        grad_accum_steps = self.config['training'].get('gradient_accumulation_steps', 1)

        with autocast('cuda', enabled=self.use_mixed_precision, dtype=torch.float16):
            outputs = self.model(input_ids, labels=labels)
            loss = outputs['loss'] / grad_accum_steps

        if self.use_mixed_precision:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        if not is_accumulating:
            if self.use_mixed_precision:
                self.scaler.unscale_(self.optimizer)

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training']['grad_clip']
            )

            if self.use_mixed_precision:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad()

        return loss.item() * grad_accum_steps
    
    def train(self, dataloader, num_steps):
        """Main training loop"""
        
        progress_bar = None
        if self.is_main:
            progress_bar = tqdm(total=num_steps, desc="Training")
        
        running_loss = 0.0
        log_interval = self.config['training']['log_every']
        save_interval = self.config['training'].get('save_every', 1000)
        grad_accum_steps = self.config['training'].get('gradient_accumulation_steps', 1)

        for epoch in range(100):  # Large number
            for batch_idx, batch in enumerate(dataloader):
                is_accumulating = (batch_idx + 1) % grad_accum_steps != 0
                loss = self.train_step(batch, is_accumulating=is_accumulating)
                running_loss += loss
                self.current_loss = loss

                if is_accumulating:
                    continue

                self.step += 1

                # Logging
                if self.step % log_interval == 0 and self.is_main:
                    avg_loss = running_loss / (log_interval * grad_accum_steps)
                    progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
                    running_loss = 0.0

                # Save checkpoint
                if self.step % save_interval == 0 and self.is_main:
                    from src.utils.checkpoint import save_checkpoint
                    save_checkpoint(
                        self.model,
                        self.optimizer,
                        self.step,
                        self.current_loss,
                        self.config['training'].get('output_dir', 'experiments/run'),
                        self.rank
                    )

                # Update progress
                if self.is_main:
                    progress_bar.update(1)

                # Check if done
                if self.step >= num_steps:
                    if self.is_main:
                        progress_bar.close()
                    return
        
        if self.is_main:
            progress_bar.close()
