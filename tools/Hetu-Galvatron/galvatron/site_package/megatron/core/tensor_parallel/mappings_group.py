import torch

from megatron.core.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_group,
)
from .utils import split_tensor_along_last_dim


def _reduce(input_, group):
    """All-reduce the input tensor across model parallel group."""

    # Bypass the function if we are using only 1 GPU.
    if get_tensor_model_parallel_world_size_group(group)==1:
        return input_

    # All-reduce.
    torch.distributed.all_reduce(input_, group=group)

    return input_


def _split_along_last_dim(input_, group):
    """Split the tensor along its last dimension and keep the
    corresponding slice."""

    world_size = get_tensor_model_parallel_world_size_group(group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Split along last dimension.
    input_list = split_tensor_along_last_dim(input_, world_size)

    # Note: torch.split does not create contiguous tensors by default.
    rank = get_tensor_model_parallel_rank_group(group)
    output = input_list[rank].contiguous()

    return output


def _split_along_first_dim(input_, group):
    """Split the tensor along its first dimension and keep the
    corresponding slice."""

    world_size = get_tensor_model_parallel_world_size_group(group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Split along first dimension.
    dim_size = input_.size()[0]
    assert dim_size % world_size == 0, \
        "First dimension of the tensor should be divisible by tensor parallel size"
    local_dim_size = dim_size // world_size
    rank = get_tensor_model_parallel_rank_group(group)
    dim_offset = rank * local_dim_size

    output = input_[dim_offset:dim_offset+local_dim_size].contiguous()

    return output


def _gather_along_last_dim(input_, group):
    """Gather tensors and concatinate along the last dimension."""

    world_size = get_tensor_model_parallel_world_size_group(group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Size and dimension.
    last_dim = input_.dim() - 1
    rank = get_tensor_model_parallel_rank_group(group)

    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    torch.distributed.all_gather(tensor_list, input_, group=group)

    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=last_dim).contiguous()

    return output


def _gather_along_first_dim(input_, group):
    """Gather tensors and concatinate along the first dimension."""

    world_size = get_tensor_model_parallel_world_size_group(group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size

    output = torch.empty(dim_size, dtype=input_.dtype,
                         device=torch.cuda.current_device())
    torch.distributed._all_gather_base(output, input_.contiguous(),
                                       group=group)

    return output

def _reduce_scatter_along_first_dim(input_, group):
    """Reduce-scatter the input tensor across model parallel group."""
    world_size = get_tensor_model_parallel_world_size_group(group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    assert dim_size[0] % world_size == 0, \
        "First dimension of the tensor should be divisible by tensor parallel size"
    
    dim_size[0] = dim_size[0] // world_size
   
    output = torch.empty(dim_size, dtype=input_.dtype,
                         device=torch.cuda.current_device())
    torch.distributed._reduce_scatter_base(output, input_.contiguous(), 
                                           group=group)
    return output


class _CopyToModelParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return input_
    
    @staticmethod
    def forward(ctx, input_, group):
        ctx.group = group
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(grad_output, ctx.group), None


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-reduce the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_, group):
        return _reduce(input_, group)
    
    @staticmethod
    def forward(ctx, input_, group):
        return _reduce(input_, group)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class _ScatterToModelParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def symbolic(graph, input_, group):
        return _split_along_last_dim(input_, group)

    @staticmethod
    def forward(ctx, input_, group):
        ctx.group = group
        return _split_along_last_dim(input_, group)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather_along_last_dim(grad_output, ctx.group)


class _GatherFromModelParallelRegion(torch.autograd.Function):
    """Gather the input from model parallel region and concatinate."""

    @staticmethod
    def symbolic(graph, input_, group):
        return _gather_along_last_dim(input_, group)
    
    @staticmethod
    def forward(ctx, input_, group):
        ctx.group = group
        return _gather_along_last_dim(input_, group)

    @staticmethod
    def backward(ctx, grad_output):
        return _split_along_last_dim(grad_output, ctx.group), None


class _ScatterToSequenceParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def symbolic(graph, input_, group):
        return _split_along_first_dim(input_, group)

    @staticmethod
    def forward(ctx, input_, group):
        ctx.group = group
        return _split_along_first_dim(input_, group)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather_along_first_dim(grad_output, ctx.group)


class _GatherFromSequenceParallelRegion(torch.autograd.Function):
    """Gather the input from sequence parallel region and concatinate.""" 

    @staticmethod
    def symbolic(graph, input_, group, tensor_parallel_output_grad=True):
        return _gather_along_first_dim(input_, group)
    
    @staticmethod
    def forward(ctx, input_, group, tensor_parallel_output_grad=True):
        ctx.group = group
        ctx.tensor_parallel_output_grad = tensor_parallel_output_grad
        return _gather_along_first_dim(input_, group)

    @staticmethod
    def backward(ctx, grad_output):
        tensor_parallel_output_grad = ctx.tensor_parallel_output_grad

        # If the computation graph after the gather operation is
        # in the tensor parallel mode, output gradients need to reduce 
        # scattered and whereas if the computation is duplicated, 
        # output gradients need to be scattered.
        if tensor_parallel_output_grad:
            return _reduce_scatter_along_first_dim(grad_output, ctx.group), None
        else:
            return _split_along_first_dim(grad_output, ctx.group), None


class _ReduceScatterToSequenceParallelRegion(torch.autograd.Function):
    """Reduce scatter the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_, group):
        return _reduce_scatter_along_first_dim(input_, group)
    
    @staticmethod
    def forward(ctx, input_, group):
        ctx.group = group
        return _reduce_scatter_along_first_dim(input_, group)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather_along_first_dim(grad_output, ctx.group)


# -----------------
# Helper functions.
# -----------------

def copy_to_tensor_model_parallel_region_group(input_, group):
    return _CopyToModelParallelRegion.apply(input_, group)


def reduce_from_tensor_model_parallel_region_group(input_, group):
    return _ReduceFromModelParallelRegion.apply(input_, group)


def scatter_to_tensor_model_parallel_region_group(input_, group):
    return _ScatterToModelParallelRegion.apply(input_, group)


def gather_from_tensor_model_parallel_region_group(input_, group):
    return _GatherFromModelParallelRegion.apply(input_, group)


def scatter_to_sequence_parallel_region_group(input_, group):
    return _ScatterToSequenceParallelRegion.apply(input_, group)


def gather_from_sequence_parallel_region_group(input_, group, tensor_parallel_output_grad=True):
    return _GatherFromSequenceParallelRegion.apply(input_, group, tensor_parallel_output_grad)


def reduce_scatter_to_sequence_parallel_region_group(input_, group):
    return _ReduceScatterToSequenceParallelRegion.apply(input_, group)

# -----------------
# tensor parallel communication group util functions
# -----------------

def get_tensor_model_parallel_world_size_group(group):
    """Return world size for the tensor model parallel group."""
    return torch.distributed.get_world_size(group=group)

def get_tensor_model_parallel_rank_group(group):
    """Return my rank for the tensor model parallel group."""
    return torch.distributed.get_rank(group=group)

