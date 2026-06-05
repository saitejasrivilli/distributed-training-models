import torch
import torch.nn as nn
import torch.distributed as dist
from typing import List, Tuple


class PipelineStage(nn.Module):
    """Single stage of pipeline parallelism (subset of model layers)"""

    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers

    def forward(self, x, labels=None):
        for layer in self.layers:
            x = layer(x)
        return x


class PipelineParallelModel(nn.Module):
    """Wraps model with pipeline parallelism (layer-wise model sharding)"""

    def __init__(self, model, world_size=1, rank=0):
        super().__init__()
        self.model = model
        self.world_size = world_size
        self.rank = rank
        self.device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

        if world_size == 1:
            self.stage = self.model
            return

        # Split transformer blocks across stages
        if hasattr(model, 'blocks'):
            num_blocks = len(model.blocks)
            blocks_per_stage = num_blocks // world_size
            remainder = num_blocks % world_size

            start_idx = rank * blocks_per_stage + min(rank, remainder)
            end_idx = start_idx + blocks_per_stage + (1 if rank < remainder else 0)

            # Create stage with assigned blocks
            stage_blocks = nn.ModuleList(model.blocks[start_idx:end_idx])
            self.stage = PipelineStage(stage_blocks)
            self.start_block_idx = start_idx
            self.end_block_idx = end_idx
        else:
            self.stage = model

    def forward(self, x, labels=None):
        if self.world_size == 1:
            return self.model(x, labels=labels)

        x = x.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)

        batch_size, seq_len = x.shape[0], x.shape[1]
        d_model = self.model.config.d_model

        if self.rank == 0:
            x = self.model.token_embed(x)
            pos = torch.arange(0, seq_len, dtype=torch.long, device=self.device).unsqueeze(0)
            x = x + self.model.pos_embed(pos)
            x = self.model.dropout(x)
        else:
            # Receive activations from the previous stage
            x = torch.empty(batch_size, seq_len, d_model, device=self.device)
            if dist.is_available() and dist.is_initialized():
                dist.recv(x, src=self.rank - 1)

        x = self.stage(x)

        if self.rank == self.world_size - 1:
            x = self.model.ln_f(x)
            logits = self.model.lm_head(x)

            loss = None
            if labels is not None:
                import torch.nn.functional as F
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100
                )

            return {"logits": logits, "loss": loss}
        else:
            if dist.is_available() and dist.is_initialized():
                dist.send(x, dst=self.rank + 1)
            return {"logits": None, "loss": None}

    def get_device(self):
        return self.device


def split_model_pipeline(model, world_size=1, rank=0):
    """Wrap model with pipeline parallelism"""
    if world_size == 1:
        return model

    return PipelineParallelModel(model, world_size=world_size, rank=rank)


class PipelineParallelTrainer:
    """Trainer that handles pipeline parallel communication"""

    def __init__(self, base_trainer, world_size=1, rank=0):
        self.trainer = base_trainer
        self.world_size = world_size
        self.rank = rank
        self.device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

    def pipeline_forward(self, batch):
        """Execute forward pass with pipeline communication"""
        if self.world_size == 1:
            return self.trainer.train_step(batch)

        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device) if 'labels' in batch else None

        # Forward pass through assigned stage
        if self.rank == 0:
            x = self.trainer.model.model.token_embed(input_ids)
            pos = torch.arange(0, x.shape[1], dtype=torch.long, device=x.device).unsqueeze(0)
            x = x + self.trainer.model.model.pos_embed(pos)
            x = self.trainer.model.model.dropout(x)
        else:
            # Receive from previous stage
            d_model = self.trainer.model.model.config.d_model
            x = torch.empty(input_ids.shape[0], input_ids.shape[1], d_model, device=self.device)
            if dist.is_available() and dist.is_initialized():
                dist.recv(x, src=self.rank - 1)

        # Local stage computation
        x = self.trainer.model.stage(x)

        # Send to next stage or compute loss
        if self.rank < self.world_size - 1:
            if dist.is_available() and dist.is_initialized():
                dist.send(x, dst=self.rank + 1)
            return 0.0  # Dummy loss for intermediate stages
        else:
            x = self.trainer.model.model.ln_f(x)
            logits = self.trainer.model.model.lm_head(x)

            import torch.nn.functional as F
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )

            return loss.item()
