import torch
import torch.nn as nn
import torch.distributed as dist


class ParallelLinear(nn.Module):
    """Column-wise tensor parallel linear layer.

    Splits weight matrix across GPUs: [out_features, in_features] -> [out_features/world_size, in_features]
    Forward: all_gather input, local matmul, reduce_scatter output
    Backward: all_gather gradient, local matmul, reduce_scatter weight gradient
    """

    def __init__(self, in_features, out_features, bias=True, world_size=1, rank=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.world_size = world_size
        self.rank = rank

        assert out_features % world_size == 0, f"out_features ({out_features}) must be divisible by world_size ({world_size})"

        self.out_features_per_rank = out_features // world_size

        self.weight = nn.Parameter(torch.empty(self.out_features_per_rank, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features_per_rank))
        else:
            self.register_parameter('bias', None)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.weight, a=0, mode='fan_in', nonlinearity='linear')
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.world_size == 1:
            return torch.nn.functional.linear(x, self.weight, self.bias)

        # All-gather inputs
        x_shape = x.shape
        x_flat = x.view(-1, x.shape[-1])

        if dist.is_available() and dist.is_initialized():
            x_gather = [torch.empty_like(x_flat) for _ in range(self.world_size)]
            dist.all_gather(x_gather, x_flat)
            x_gather = torch.cat(x_gather, dim=0)
        else:
            x_gather = x_flat

        # Local linear transformation
        out = torch.nn.functional.linear(x_gather, self.weight, self.bias)

        # Each rank takes its local-batch slice of the full output
        if dist.is_available() and dist.is_initialized():
            out_scatter_list = list(torch.chunk(out, self.world_size, dim=0))
            out_scatter = out_scatter_list[self.rank].contiguous()
        else:
            out_scatter = out

        return out_scatter.view(*x_shape[:-1], -1)


def convert_linear_to_tensor_parallel(model, world_size=1, rank=0):
    """Convert all linear layers to tensor parallel versions"""
    if world_size == 1:
        return model

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]

            parent = model
            for part in parent_name.split('.'):
                if part:
                    parent = getattr(parent, part)

            parallel_linear = ParallelLinear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                world_size=world_size,
                rank=rank
            )

            with torch.no_grad():
                if world_size > 1:
                    start_idx = rank * module.out_features // world_size
                    end_idx = (rank + 1) * module.out_features // world_size
                    parallel_linear.weight.copy_(module.weight[start_idx:end_idx])
                    if module.bias is not None:
                        parallel_linear.bias.copy_(module.bias[start_idx:end_idx])
                else:
                    parallel_linear.weight.copy_(module.weight)
                    if module.bias is not None:
                        parallel_linear.bias.copy_(module.bias)

            setattr(parent, child_name, parallel_linear)

    return model
