import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import CPUOffload, ShardingStrategy
import torch.distributed as dist
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import os
import time
import logging

logger = logging.getLogger(__name__)

# A30 PCIe theoretical peak: 165 TFLOPS FP16
_A30_PEAK_FLOPS_FP16 = 165e12


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

        # MFU tracking
        self.mfu_history: list[float] = []
        
    def compute_mfu(
        self,
        batch_size: int,
        seq_len: int,
        elapsed_seconds: float,
        peak_flops: float = _A30_PEAK_FLOPS_FP16,
    ) -> float:
        """
        Model FLOP Utilisation (MFU).

        achieved_flops = 6 * N * B * S / elapsed_seconds
          N = total model parameters
          B = batch_size * world_size  (global batch)
          S = sequence length
          factor 6 = 2 (fused multiply-add) * 3 (forward + backward + optimizer)
        """
        N = sum(p.numel() for p in self.model.parameters())
        B = batch_size * self.world_size
        achieved = (6 * N * B * seq_len) / elapsed_seconds
        mfu = achieved / peak_flops
        self.mfu_history.append(mfu)
        return mfu

    def get_mfu_summary(self) -> dict:
        """Return mean / peak / min MFU over all logged steps."""
        if not self.mfu_history:
            return {"mean": 0.0, "peak": 0.0, "min": 0.0}
        return {
            "mean": sum(self.mfu_history) / len(self.mfu_history),
            "peak": max(self.mfu_history),
            "min": min(self.mfu_history),
        }

    def save_checkpoint(self, path: str, step: int, loss: float) -> None:
        """Save model + optimiser state to *path*."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        # Unwrap DDP / FSDP so we save raw state_dict
        raw_model = self.model
        if isinstance(raw_model, (DDP, FSDP)):
            raw_model = raw_model.module
        torch.save(
            {
                "step": step,
                "loss": loss,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
                "config": self.config,
            },
            path,
        )
        if self.is_main:
            logger.info("Checkpoint saved: %s (step=%d, loss=%.4f)", path, step, loss)

    def load_checkpoint(self, path: str) -> int:
        """
        Load checkpoint from *path* into model / optimiser.
        Returns the step to resume from.
        """
        ckpt = torch.load(path, map_location=self.device)
        raw_model = self.model
        if isinstance(raw_model, (DDP, FSDP)):
            raw_model = raw_model.module
        raw_model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if self.scaler and ckpt.get("scaler_state_dict"):
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.step = ckpt["step"]
        self.current_loss = ckpt["loss"]
        if self.is_main:
            logger.info("Resumed from checkpoint: %s (step=%d)", path, self.step)
        return self.step

    def train_with_recovery(
        self,
        dataloader,
        num_steps: int,
        checkpoint_dir: str,
        save_every: int = 500,
    ) -> None:
        """
        Training loop with automatic checkpointing and CUDA-error recovery.

        Saves a checkpoint every *save_every* optimiser steps.
        On CUDA OOM or runtime error, catches the exception, logs the failure,
        and re-raises with instructions to resume from the last checkpoint.
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        last_ckpt_path: str | None = None

        log_interval = self.config['training']['log_every']
        grad_accum_steps = self.config['training'].get('gradient_accumulation_steps', 1)

        progress_bar = None
        if self.is_main:
            progress_bar = tqdm(total=num_steps, initial=self.step, desc="Training (with recovery)")

        try:
            for epoch in range(100):
                for batch_idx, batch in enumerate(dataloader):
                    is_accumulating = (batch_idx + 1) % grad_accum_steps != 0

                    t0 = time.perf_counter()
                    loss = self.train_step(batch, is_accumulating=is_accumulating)
                    elapsed = time.perf_counter() - t0

                    if is_accumulating:
                        continue

                    self.step += 1
                    self.current_loss = loss

                    if self.step % log_interval == 0 and self.is_main:
                        input_ids = batch['input_ids']
                        batch_size = input_ids.size(0)
                        seq_len = input_ids.size(1)
                        mfu = self.compute_mfu(batch_size, seq_len, elapsed * grad_accum_steps)
                        tokens_per_sec = (batch_size * self.world_size * seq_len * log_interval) / (elapsed * log_interval)
                        logger.info(
                            "step=%d loss=%.4f mfu=%.2f%% tok/s=%.0f",
                            self.step, loss, mfu * 100, tokens_per_sec,
                        )
                        if progress_bar:
                            progress_bar.set_postfix({
                                "loss": f"{loss:.4f}",
                                "mfu": f"{mfu:.2%}",
                            })

                    if self.step % save_every == 0 and self.is_main:
                        ckpt_path = os.path.join(checkpoint_dir, f"ckpt_step{self.step}.pt")
                        self.save_checkpoint(ckpt_path, self.step, loss)
                        last_ckpt_path = ckpt_path

                    if progress_bar:
                        progress_bar.update(1)

                    if self.step >= num_steps:
                        if progress_bar:
                            progress_bar.close()
                        return

        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            msg = (
                f"Training failed at step {self.step} with: {exc}\n"
                f"To resume, call load_checkpoint('{last_ckpt_path or '<last_ckpt>'}') "
                "before calling train_with_recovery again."
            )
            logger.error(msg)
            if progress_bar:
                progress_bar.close()
            raise RuntimeError(msg) from exc

        if progress_bar:
            progress_bar.close()

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

        # Accumulate elapsed time over log_interval steps for MFU calculation
        _step_start: float = time.perf_counter()
        _tokens_since_log: int = 0

        for epoch in range(100):  # Large number
            for batch_idx, batch in enumerate(dataloader):
                is_accumulating = (batch_idx + 1) % grad_accum_steps != 0

                t0 = time.perf_counter()
                loss = self.train_step(batch, is_accumulating=is_accumulating)
                elapsed_step = time.perf_counter() - t0

                running_loss += loss
                self.current_loss = loss

                if is_accumulating:
                    continue

                self.step += 1

                input_ids = batch['input_ids']
                batch_size = input_ids.size(0)
                seq_len = input_ids.size(1)
                _tokens_since_log += batch_size * self.world_size * seq_len

                # Logging
                if self.step % log_interval == 0 and self.is_main:
                    avg_loss = running_loss / (log_interval * grad_accum_steps)
                    elapsed_interval = time.perf_counter() - _step_start
                    mfu = self.compute_mfu(batch_size, seq_len, elapsed_step * grad_accum_steps)
                    tokens_per_sec = _tokens_since_log / max(elapsed_interval, 1e-9)
                    print(
                        f"step={self.step} loss={avg_loss:.4f} "
                        f"mfu={mfu:.2%} tok/s={tokens_per_sec:.0f}"
                    )
                    progress_bar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'mfu': f'{mfu:.2%}',
                        'tok/s': f'{tokens_per_sec:.0f}',
                    })
                    running_loss = 0.0
                    _step_start = time.perf_counter()
                    _tokens_since_log = 0

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
