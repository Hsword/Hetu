import torch

def _split_along_first_dim(input_, group):
    """Split the tensor along its first dimension and keep the
    corresponding slice."""

    world_size = torch.distributed.get_world_size(group=group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Split along first dimension.
    dim_size = input_.size()[0]
    assert dim_size % world_size == 0, \
        "First dimension of the tensor should be divisible by tensor parallel size"
    local_dim_size = dim_size // world_size
    rank = torch.distributed.get_rank(group=group)
    dim_offset = rank * local_dim_size

    output = input_[dim_offset:dim_offset+local_dim_size].contiguous()

    return output

def _gather_along_first_dim(input_, group):
    """Gather tensors and concatinate along the first dimension."""

    world_size = torch.distributed.get_world_size(group=group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size

    output = torch.empty(dim_size, dtype=input_.dtype,
                         device=torch.cuda.current_device())
    torch.distributed.all_gather_into_tensor(output, input_.contiguous(),
                                       group=group)

    return output

class _Split(torch.autograd.Function):
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
        return _gather_along_first_dim(grad_output, ctx.group), None

class _Gather(torch.autograd.Function):
    """Gather the input from model parallel region and concatinate."""

    @staticmethod
    def symbolic(graph, input_):
        return _gather_along_first_dim(input_)
    
    @staticmethod
    def forward(ctx, input_, group):
        ctx.group = group
        return _gather_along_first_dim(input_, group)

    @staticmethod
    def backward(ctx, grad_output):
        return _split_along_first_dim(grad_output, ctx.group), None

def split_to_group(input_, group):
    return _Split.apply(input_, group)

def gather_from_group(input_, group):
    return _Gather.apply(input_, group)