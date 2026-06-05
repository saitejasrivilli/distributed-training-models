import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class _ColumnParallelAllGather(torch.autograd.Function):
    """Gradient-aware all_gather for column-parallel linear.

    Forward : cat([local_out_0, ..., local_out_N], dim=-1)  →  full output
    Backward: slice grad_output to get local grad; all_reduce is handled by the
              input hook registered in ParallelLinear.forward so that grad_x is
              summed across all ranks before flowing to earlier layers.
    """

    @staticmethod
    def forward(ctx, tensor, world_size, rank):
        ctx.world_size = world_size
        ctx.rank = rank
        gather_list = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(gather_list, tensor.contiguous())
        return torch.cat(gather_list, dim=-1)

    @staticmethod
    def backward(ctx, grad_output):
        rank = ctx.rank
        world_size = ctx.world_size
        shard_size = grad_output.shape[-1] // world_size
        grad_local = grad_output[..., rank * shard_size:(rank + 1) * shard_size].contiguous()
        return grad_local, None, None


class ParallelLinear(nn.Module):
    """Column-wise tensor parallel linear layer.

    Each rank holds weight[out/N * rank : out/N * (rank+1), in_features].
    Forward : local matmul  →  all_gather  →  full [*, out_features] output.
    Backward: grad_output sliced to local shard; input gradient all_reduced
              across ranks so earlier layers see the correct summed gradient.
    """

    def __init__(self, in_features, out_features, bias=True, world_size=1, rank=0):
        super().__init__()
        assert out_features % world_size == 0, (
            f"out_features ({out_features}) must be divisible by world_size ({world_size})"
        )
        self.in_features = in_features
        self.out_features = out_features
        self.world_size = world_size
        self.rank = rank
        self.out_per_rank = out_features // world_size

        self.weight = nn.Parameter(torch.empty(self.out_per_rank, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_per_rank))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.weight, a=0, mode='fan_in', nonlinearity='linear')
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.world_size == 1:
            return F.linear(x, self.weight, self.bias)

        # Register all_reduce hook on x so grad_x is correctly summed across ranks
        if x.requires_grad and dist.is_initialized():
            def _allreduce_grad(grad):
                dist.all_reduce(grad, op=dist.ReduceOp.SUM)
                return grad
            x.register_hook(_allreduce_grad)

        # Column-parallel: each rank computes its output shard
        local_out = F.linear(x, self.weight, self.bias)  # [*, out/N]

        # All-gather to reconstruct full output with gradient support
        return _ColumnParallelAllGather.apply(local_out, self.world_size, self.rank)


def convert_linear_to_tensor_parallel(model, world_size=1, rank=0, skip_modules=None):
    """Replace nn.Linear layers with ParallelLinear, skipping named modules.

    skip_modules: set of module names to leave as-is (e.g. {'lm_head'} whose
    vocab dimension 50257 is not divisible by common world_sizes).
    """
    if world_size == 1:
        return model

    if skip_modules is None:
        skip_modules = set()

    replacements = {}
    for name, module in model.named_modules():
        if name in skip_modules:
            continue
        if not isinstance(module, nn.Linear):
            continue
        if module.out_features % world_size != 0:
            continue  # skip layers whose output dim isn't divisible

        replacements[name] = module

    for name, module in replacements.items():
        parent_name, child_name = name.rsplit('.', 1) if '.' in name else ('', name)
        parent = model
        for part in parent_name.split('.'):
            if part:
                parent = getattr(parent, part)

        target_device = module.weight.device
        pl = ParallelLinear(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            world_size=world_size,
            rank=rank,
        ).to(target_device)
        with torch.no_grad():
            start = rank * module.out_features // world_size
            end = (rank + 1) * module.out_features // world_size
            pl.weight.copy_(module.weight[start:end])
            if module.bias is not None:
                pl.bias.copy_(module.bias[start:end])

        setattr(parent, child_name, pl)

    return model
