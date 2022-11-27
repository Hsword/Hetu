import torch

def gather_from_tensor_model_parallel_region_group(input_, group):
    return _GatherFromModelParallelRegion.apply(input_, group)

def _split(input_, group):
    """Split the tensor along its first dimension and keep the
    corresponding slice."""

    world_size = get_tensor_model_parallel_world_size_group(group)
    # Bypass the function if we are using only 1 GPU.
    if world_size==1:
        return input_

    # Split along last dimension.
    input_list = split_tensor_along_first_dim(input_, world_size)

    # Note: torch.split does not create contiguous tensors by default.
    rank = get_tensor_model_parallel_rank_group(group)
    output = input_list[rank].contiguous()

    return output

def _gather(input_, group):
    """Gather tensors and concatinate along the first dimension."""

    world_size = get_tensor_model_parallel_world_size_group(group)
    # Bypass the function if we are using only 1 GPU.
    if world_size==1:
        return input_

    # Size and dimension.
    first_dim = 0
    rank = get_tensor_model_parallel_rank_group(group)

    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    torch.distributed.all_gather(tensor_list, input_, group=group)

    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=first_dim).contiguous()

    return output

class _GatherFromModelParallelRegion(torch.autograd.Function):
    """Gather the input from model parallel region and concatinate."""

    @staticmethod
    def symbolic(graph, input_):
        return _gather(input_)
    
    @staticmethod
    def forward(ctx, input_, group):
        ctx.group = group
        return _gather(input_, group)

    @staticmethod
    def backward(ctx, grad_output):
        return _split(grad_output, ctx.group), None

# -----------------
# tensor parallel communication group util functions
# -----------------

def get_tensor_model_parallel_world_size_group(group):
    """Return world size for the tensor model parallel group."""
    return torch.distributed.get_world_size(group=group)

def get_tensor_model_parallel_rank_group(group):
    """Return my rank for the tensor model parallel group."""
    return torch.distributed.get_rank(group=group)

# -----------------
# tensor split utils
# -----------------

def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, '{} is not divisible by {}'.format(
        numerator, denominator)

def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator

def split_tensor_along_first_dim(tensor, num_partitions,
                                contiguous_split_chunks=False):
    """Split a tensor along its first dimension.
    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.
    """
    # Get the size and dimension.
    first_dim = 0
    first_dim_size = divide(tensor.size()[first_dim], num_partitions)
    # Split.
    tensor_list = torch.split(tensor, first_dim_size, dim=first_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list