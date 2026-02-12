import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
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
        self.current_loss = 0.0  # Track current loss
        
    def train_step(self, batch):
        """Single training step"""
        self.model.train()
        
        input_ids = batch['input_ids'].cuda()
        labels = batch['labels'].cuda()
        
        # Forward pass
        outputs = self.model(input_ids, labels=labels)
        loss = outputs['loss']
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config['training']['grad_clip']
        )
        
        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return loss.item()
    
    def train(self, dataloader, num_steps):
        """Main training loop"""
        
        progress_bar = None
        if self.is_main:
            progress_bar = tqdm(total=num_steps, desc="Training")
        
        running_loss = 0.0
        log_interval = self.config['training']['log_every']
        save_interval = self.config['training'].get('save_every', 1000)
        
        for epoch in range(100):  # Large number
            for batch_idx, batch in enumerate(dataloader):
                
                loss = self.train_step(batch)
                running_loss += loss
                self.current_loss = loss  # Update current loss
                
                self.step += 1
                
                # Logging
                if self.step % log_interval == 0 and self.is_main:
                    avg_loss = running_loss / log_interval
                    progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
                    running_loss = 0.0
                
                # Save checkpoint
                if self.step % save_interval == 0 and self.is_main:
                    from src.utils.checkpoint import save_checkpoint
                    save_checkpoint(
                        self.model.module if self.world_size > 1 else self.model,
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
