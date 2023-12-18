from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import ShardingStrategy, CPUOffload, MixedPrecision, BackwardPrefetch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper, CheckpointImpl
from torch.distributed.fsdp.wrap import _recursive_wrap, lambda_auto_wrap_policy
import torch.nn as nn
import torch
from typing import Tuple, List
from functools import partial

def wrap_data_parallel(module, dp_type = None, dp_group = None, module_type='bert_enc', pp_device = None, mixed_precision=torch.bfloat16, pp_on=False, wrap_block_name=None):
    if dp_type is None:
        return module
    else:
        assert pp_device is not None
        from .arguments import get_args
        fsdp_type_dict = {0:get_args().default_dp_type, 1:'zero3'}
        assert dp_type in fsdp_type_dict.keys()
        return wrap_module_fsdp_manually(module, pp_device, module_type, dp_group, fsdp_type=fsdp_type_dict[dp_type], mixed_precision=mixed_precision, pp_on=pp_on, wrap_block_name=wrap_block_name)

def param_init_fn(module):
    module.to_empty(device=torch.device("cuda"))
    for m in module.modules():
        if callable(getattr(m, 'reset_parameters', None)):
            m.reset_parameters()

def wrap_module_fsdp_manually(module, pp_device, module_type='bert_enc', dp_group=None, fsdp_type='zero3', mixed_precision=torch.bfloat16, pp_on=False, wrap_block_name=None):
    comm_group = None if dp_group is None else dp_group.group
    sharding_strategy = {'ddp': ShardingStrategy.NO_SHARD,
                           'zero2': ShardingStrategy.SHARD_GRAD_OP,
                           'zero3': ShardingStrategy.FULL_SHARD}[fsdp_type]
    mixed_precision_policy = MixedPrecision(
        param_dtype=mixed_precision, # Param precision
        reduce_dtype=mixed_precision, # Gradient communication precision
        buffer_dtype=mixed_precision, # Buffer precision
        cast_forward_inputs=True,
        cast_root_forward_inputs=True,
    )
    from .arguments import get_args
    args = get_args()
    backward_prefetch = None if pp_on else BackwardPrefetch.BACKWARD_PRE
    fsdp_args = dict(process_group = comm_group, 
                    sharding_strategy = sharding_strategy, 
                    mixed_precision=mixed_precision_policy, 
                    backward_prefetch=backward_prefetch,
                    device_id=pp_device,
                    param_init_fn=param_init_fn if 'initialize_on_meta' in args and args.initialize_on_meta else None,
                    limit_all_gathers=True)

    # Wrap given block
    if wrap_block_name is not None:
        if 'enc' in module_type or 'dec' in module_type:
            module = apply_fsdp(module, fsdp_args, wrap_block_name)
        else: 
            # return module
            if 'initialize_on_meta' in args and args.initialize_on_meta:
                module = FSDP(module, **fsdp_args)
            else:
                module = FSDP(module.to(pp_device), **fsdp_args)
        return module

    # Wrap manually
    if module_type in ['bert_enc', 'vit_enc']:
        sub_module = module.module.layer[0]
        setattr(sub_module, 'attention', FSDP(sub_module.attention, **fsdp_args))
        setattr(sub_module, 'mlp', FSDP(sub_module.mlp, **fsdp_args))
        return FSDP(module, **fsdp_args)
    elif module_type in ['swin_enc']:
        sub_module = module.module.block
        setattr(sub_module, 'attention', FSDP(sub_module.attention, **fsdp_args))
        setattr(sub_module, 'intermediate', FSDP(sub_module.intermediate, **fsdp_args))
        return FSDP(module, **fsdp_args)
    elif module_type in ['t5_enc']:
        sub_module = module.module.block.t5_block
        setattr(sub_module.layer[0], 'SelfAttention', FSDP(sub_module.layer[0].SelfAttention.cuda(pp_device), **fsdp_args))
        sub_module.layer[-1] = FSDP(sub_module.layer[-1].cuda(pp_device), **fsdp_args)
        return FSDP(module, **fsdp_args)
    elif module_type in ['t5_dec']:
        module_ = module.module
        sub_module = module_.block.t5_block
        setattr(module_, 'block', FSDP(module_.block.cuda(pp_device), **fsdp_args))
        setattr(sub_module.layer[0], 'SelfAttention', FSDP(sub_module.layer[0].SelfAttention.cuda(pp_device), **fsdp_args))
        setattr(sub_module.layer[1], 'EncDecAttention', FSDP(sub_module.layer[1].EncDecAttention.cuda(pp_device), **fsdp_args))
        sub_module.layer[-1] = FSDP(sub_module.layer[-1].cuda(pp_device), **fsdp_args)
        return FSDP(module, **fsdp_args)
    elif module_type in ['gpt_dec']:
        module.module.layers[0] = FSDP(module.module.layers[0], **fsdp_args)
        return FSDP(module, **fsdp_args)
    else:
        if 'initialize_on_meta' in args and args.initialize_on_meta:
            return FSDP(module, **fsdp_args)
        else:
            return FSDP(module.to(pp_device), **fsdp_args)

def apply_fsdp(model, fsdp_args, wrap_block_name):
    check_fn=lambda submodule: (any(isinstance(submodule, block) for block in wrap_block_name))
    _recursive_wrap(
        module=model,
        auto_wrap_policy=partial(lambda_auto_wrap_policy, lambda_fn=check_fn),
        wrapper_cls=FSDP,
        ignored_modules=set(),
        ignored_params=set(),
        only_wrap_children=True,
        **fsdp_args
    )
    return model

def apply_ckpt(model, checkpoint_wrapper_fn, wrap_block_name):
    check_fn=lambda submodule: (any(isinstance(submodule, block) for block in wrap_block_name))
    _recursive_wrap(
        module=model,
        auto_wrap_policy=partial(lambda_auto_wrap_policy, lambda_fn=check_fn),
        wrapper_cls=checkpoint_wrapper_fn,
        ignored_modules=set(),
        ignored_params=set(),
        only_wrap_children=True
    )
    return model

def wrap_modules_checkpoint(module_list, checkpoint_flags, wrap_block_name=None):
    m = module_list
    if isinstance(m, FSDP):
        m = m._fsdp_wrapped_module
    assert len(m) == len(checkpoint_flags)
    for i in range(len(m)):
        if checkpoint_flags[i]:
            if wrap_block_name is not None:
                m[i] = apply_ckpt(m[i], checkpoint_wrapper, wrap_block_name)
            else:
                m[i] = checkpoint_wrapper(m[i])
    return module_list

def wrap_model_checkpoint(model, wrap_block_names=[]):
    model_ = model._fsdp_wrapped_module if isinstance(model, FSDP) else model
    apply_ckpt(model_, checkpoint_wrapper, wrap_block_names)
    return model

def relocate_activations(input, allgather_group, split_group):
    if split_group is not None:
        input = split_group.split(input)
    if allgather_group is not None:
        input = allgather_group.allgather(input.contiguous())
    return input

class Module_with_relocation(nn.Module):
    def __init__(self, module, allgather_group, split_group):
        super().__init__()
        self.module = module
        self.allgather_group = allgather_group
        self.split_group = split_group
        self.relocate_activations = lambda x: relocate_activations(x, self.allgather_group, self.split_group)
        if hasattr(module, 'get_extended_attention_mask'):
            self.get_extended_attention_mask = module.get_extended_attention_mask

    def forward(self, *inputs):
        if isinstance(inputs, (Tuple, List)):
            inputs_relocated = []
            for input in inputs:
                inputs_relocated.append(self.relocate_activations(input))
            inputs_relocated = tuple(inputs_relocated)
            return self.module(*inputs_relocated)
        else:
            input_relocated = self.relocate_activations(inputs)
            return self.module(input_relocated)

def wrap_modules_data_parallel(module_list, dp_types, dp_groups, module_types, pp_devices=None, mixed_precision=torch.bfloat16, default_process_group=None, wrap_block_name=None):
    assert len(module_list) == len(dp_types)
    assert len(module_list) == len(dp_groups)
    
    process_group = default_process_group if default_process_group is not None else dp_groups[0]
    pp_on = True if process_group.size < torch.distributed.get_world_size() else False
    
    if pp_devices is not None:
        assert len(pp_devices) == len(module_list)
    for i in range(len(module_list)):
        pp_device = None if pp_devices is None else pp_devices[i]
        module_list[i] = wrap_data_parallel(module_list[i], dp_types[i], dp_groups[i], module_type=module_types[i], pp_device = pp_device, mixed_precision=mixed_precision, pp_on=pp_on, wrap_block_name=wrap_block_name)
    from .arguments import get_args
    args = get_args()
    sharding_strategy = {'ddp': ShardingStrategy.NO_SHARD,
                           'zero2': ShardingStrategy.SHARD_GRAD_OP,
                           'zero3': ShardingStrategy.FULL_SHARD}[args.default_dp_type]
    mixed_precision_policy = MixedPrecision(
        param_dtype=mixed_precision, # Param precision
        reduce_dtype=mixed_precision, # Gradient communication precision
        buffer_dtype=mixed_precision, # Buffer precision
        cast_forward_inputs=True,
        cast_root_forward_inputs=True
    )
    backward_prefetch = None if pp_on else BackwardPrefetch.BACKWARD_PRE
    fsdp_args = dict(process_group=process_group.group,
                    sharding_strategy=sharding_strategy, 
                    mixed_precision=mixed_precision_policy, 
                    backward_prefetch=backward_prefetch,
                    device_id=pp_devices[0],
                    param_init_fn=param_init_fn if 'initialize_on_meta' in args and args.initialize_on_meta else None,
                    limit_all_gathers=True)
    module_list = FSDP(module_list, **fsdp_args)
    return module_list

def wrap_model_data_parallel(model, device, wrap_block_names=[], dp_type='ddp', mixed_precision=torch.bfloat16, comm_group=None, initialize_on_meta=False, backward_prefetch=True):
    assert dp_type in ['ddp', 'zero2', 'zero3']
    sharding_strategy = {'ddp': ShardingStrategy.NO_SHARD,
                           'zero2': ShardingStrategy.SHARD_GRAD_OP,
                           'zero3': ShardingStrategy.FULL_SHARD}[dp_type]
    mixed_precision_policy = MixedPrecision(
        param_dtype=mixed_precision, # Param precision
        reduce_dtype=mixed_precision, # Gradient communication precision
        buffer_dtype=mixed_precision, # Buffer precision
        cast_forward_inputs=True,
        cast_root_forward_inputs=True,
    )
    backward_prefetch = BackwardPrefetch.BACKWARD_PRE if backward_prefetch else None
    fsdp_args = dict(process_group = comm_group, 
                    sharding_strategy = sharding_strategy, 
                    mixed_precision=mixed_precision_policy, 
                    backward_prefetch=backward_prefetch,
                    device_id=device,
                    param_init_fn=param_init_fn if initialize_on_meta else None,
                    limit_all_gathers=True)
    # Wrap specified blocks
    model = apply_fsdp(model, fsdp_args, wrap_block_names)
    # Wrap whole model
    model = FSDP(model, **fsdp_args)
    return model

def modules_to_devices(module_list, pp_devices):
    assert len(module_list) == len(pp_devices)
    for i in range(len(module_list)):
        module_list[i].to('cuda:%d'%pp_devices[i])

def wrap_modules_relocation(module_list, allgather_groups, split_groups):
    assert len(module_list) == len(allgather_groups)
    assert len(module_list) == len(split_groups)
    for i in range(len(module_list)):
        module_list[i] = Module_with_relocation(module_list[i], allgather_groups[i], split_groups[i])
    return module_list