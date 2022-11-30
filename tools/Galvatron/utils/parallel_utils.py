from torch.nn.parallel import DistributedDataParallel as DDP
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.utils.parallel import ProcessGroupName
import torch.nn as nn
import torch
from .group_comm_utils import gen_allgather_group, gen_slice_func
from typing import Tuple, List

def wrap_data_parallel(module, dp_type = None, dp_group = None, gradient_as_bucket_view = True, broadcast_buffers = True, module_type='bert_enc', pp_device = None):
    if dp_type is None:
        return module
    elif dp_type == 0:
        comm_group = None if dp_group is None else dp_group.group
        return DDP(module, process_group = comm_group, gradient_as_bucket_view=gradient_as_bucket_view, broadcast_buffers=broadcast_buffers)
    elif dp_type == 1:
        assert pp_device is not None
        return wrap_module_fsdp_manually(module, pp_device, module_type, dp_group)
    else:
        raise ValueError

def wrap_module_fsdp_manually(module, pp_device, module_type='bert_enc', dp_group=None):
    comm_group = None if dp_group is None else dp_group.group
    process_group_reduce_scatter = ProcessGroupName.reduce_scatter
    if dp_group is not None and comm_group.size() != torch.distributed.get_world_size():
        process_group_reduce_scatter = ProcessGroupName.default
    fsdp_args = dict(process_group = comm_group, process_group_reduce_scatter = process_group_reduce_scatter)

    if module_type in ['bert_enc', 'vit_enc']:
        sub_module = module.module.layer[0]
        setattr(sub_module, 'attention', FSDP(sub_module.attention.cuda(pp_device), **fsdp_args))
        setattr(sub_module, 'mlp', FSDP(sub_module.mlp.cuda(pp_device), **fsdp_args))
        return FSDP(module, **fsdp_args)
    elif module_type in ['swin_enc']:
        sub_module = module.module.block
        setattr(sub_module, 'attention', FSDP(sub_module.attention.cuda(pp_device), **fsdp_args))
        setattr(sub_module, 'intermediate', FSDP(sub_module.intermediate.cuda(pp_device), **fsdp_args))
        return FSDP(module, **fsdp_args)
    elif module_type in ['t5_enc']:
        sub_module = module.module.block.t5_block
        setattr(sub_module.layer[0], 'SelfAttention', FSDP(sub_module.layer[0].SelfAttention.cuda(pp_device), **fsdp_args))
        sub_module.layer[-1] = FSDP(sub_module.layer[-1].cuda(pp_device), **fsdp_args)
        return FSDP(module, **fsdp_args)
    elif module_type in ['t5_dec']:
        module_ = module.module
        sub_module = module.module.block.t5_block
        setattr(module_, 'block', FSDP(module_.block.cuda(pp_device), **fsdp_args))
        setattr(sub_module.layer[0], 'SelfAttention', FSDP(sub_module.layer[0].SelfAttention.cuda(pp_device), **fsdp_args))
        setattr(sub_module.layer[1], 'EncDecAttention', FSDP(sub_module.layer[1].EncDecAttention.cuda(pp_device), **fsdp_args))
        sub_module.layer[-1] = FSDP(sub_module.layer[-1].cuda(pp_device), **fsdp_args)
        return FSDP(module, **fsdp_args)
    else:
        raise NotImplementedError

def relocate_activations(input, allgather_group, slice_func):
    if allgather_group is None and slice_func is None:
        return input
    if slice_func is not None:
        input = slice_func(input)
    if allgather_group is not None:
        input = allgather_group.allgather(input.contiguous())
    return input

class Module_with_relocation(nn.Module):
    def __init__(self, module, allgather_group, slice_func):
        super().__init__()
        self.module = module
        self.allgather_group = allgather_group
        self.slice_func = slice_func
        self.relocate_activations = lambda x: relocate_activations(x, self.allgather_group, self.slice_func)
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
            return self.module(inputs_relocated)

def auto_wrap_named_module(module, dp_type, dp_group, name):
    for module_name, child in module.named_children():
        if name in module_name:
            if 'embed' in name:
                wrapped_child = wrap_data_parallel(child, dp_type, dp_group, gradient_as_bucket_view = False, broadcast_buffers = False)
            else:
                wrapped_child = wrap_data_parallel(child, dp_type, dp_group)
            setattr(module, name, wrapped_child)
        else:
            auto_wrap_named_module(child, dp_type, dp_group, name)

def my_auto_wrap(module, dp_type, dp_group):
    module_names = ['embed', 'mlp', 'attention', 'pooler', 'cls', 'layernorm']
    for name in module_names:
        auto_wrap_named_module(module, dp_type, dp_group, name)
    return module

def wrap_modules_data_parallel(module_list, dp_types, dp_groups, module_types, pp_devices=None):
    assert len(module_list) == len(dp_types)
    assert len(module_list) == len(dp_groups)
    if pp_devices is not None:
        assert len(pp_devices) == len(module_list)
    for i in range(len(module_list)):
        pp_device = None if pp_devices is None else pp_devices[i]
        # Manual Wrap
        if 'embed' in module_types[i]:
            module_list[i] = wrap_data_parallel(module_list[i], dp_types[i], dp_groups[i], gradient_as_bucket_view = False,  
                    broadcast_buffers = False, module_type=module_types[i], pp_device = pp_device)
        else:
            module_list[i] = wrap_data_parallel(module_list[i], dp_types[i], dp_groups[i], module_type=module_types[i], pp_device = pp_device)
    return module_list

def modules_to_devices(module_list, pp_devices):
    assert len(module_list) == len(pp_devices)
    for i in range(len(module_list)):
        module_list[i].to('cuda:%d'%pp_devices[i])

def wrap_modules_relocation(module_list, allgather_groups, slice_funcs):
    assert len(module_list) == len(allgather_groups)
    assert len(module_list) == len(slice_funcs)
    for i in range(len(module_list)):
        module_list[i] = Module_with_relocation(module_list[i], allgather_groups[i], slice_funcs[i])
    return module_list

def gen_label_relocation_func(input_tp_size, output_tp_size):
    allgather_group = gen_allgather_group(input_tp_size, output_tp_size, to_print = False)
    slice_func = gen_slice_func(input_tp_size, output_tp_size)
    return lambda label: relocate_activations(label, allgather_group, slice_func)
