import torch
import torch.nn as nn
import torch.distributed as dist


class PipelineParallelModel(nn.Module):
    """Synchronous pipeline parallelism (no micro-batching).

    Splits transformer blocks evenly across ranks.  Activations move forward
    via blocking send/recv.  Only the last rank has a loss, so only its local
    parameters receive gradient updates.  This reflects the real throughput
    cost of a naive synchronous pipeline: each GPU is active for 1/N of the
    wall time, creating an N-1 bubble.  Memory per GPU is ~1/N of the full
    model's layer weights.
    """

    def __init__(self, model, world_size=1, rank=0):
        super().__init__()
        self.base = model
        self.world_size = world_size
        self.rank = rank
        self.device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

        if world_size == 1:
            return

        num_blocks = len(model.blocks)
        blocks_per = num_blocks // world_size
        rem = num_blocks % world_size
        self.start_idx = rank * blocks_per + min(rank, rem)
        end_idx = self.start_idx + blocks_per + (1 if rank < rem else 0)
        self.my_blocks = nn.ModuleList(model.blocks[self.start_idx:end_idx])

    def forward(self, input_ids, labels=None):
        if self.world_size == 1:
            return self.base(input_ids, labels=labels)

        device = self.device
        B, T = input_ids.shape
        d_model = self.base.config.d_model

        if self.rank == 0:
            input_ids = input_ids.to(device)
            pos = torch.arange(T, dtype=torch.long, device=device).unsqueeze(0)
            x = self.base.dropout(
                self.base.token_embed(input_ids) + self.base.pos_embed(pos)
            )
        else:
            x = torch.empty(B, T, d_model, dtype=torch.float32, device=device)
            dist.recv(x, src=self.rank - 1)

        for block in self.my_blocks:
            x = block(x)

        if self.rank < self.world_size - 1:
            dist.send(x.contiguous(), dst=self.rank + 1)
            return {"logits": None, "loss": None}
        else:
            x = self.base.ln_f(x)
            logits = self.base.lm_head(x)
            loss = None
            if labels is not None:
                import torch.nn.functional as F
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                )
            return {"logits": logits, "loss": loss}


def split_model_pipeline(model, world_size=1, rank=0):
    if world_size == 1:
        return model
    return PipelineParallelModel(model, world_size=world_size, rank=rank)
