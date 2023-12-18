import torch
from torch import nn
import numpy as np
from galvatron.core import check_hp_config, hp_config_whole_model, get_enc_groups, mixed_precision_dtype, layer_shapes_dtypes_whole_model, get_chunks
from galvatron.core import gen_comm_groups, wrap_modules_relocation

class GalvatronModel(nn.Module):
    def __init__(self, hp_model):
        super().__init__()
        from galvatron.core import get_args
        self.args = get_args()
        self.model = hp_model
        self.iter = 0
        
    def forward_backward(self, batch, iter=None, profiler=None, loss_func=None):
        args, model = self.args, self.model
        self.iter = iter if iter is not None else self.iter
        if loss_func is not None:
            assert isinstance(batch, (tuple, list)) and isinstance(batch[0], (tuple, list)) and isinstance(batch[1], (tuple, list))
        else:
            loss_func = self.fake_loss_func
            assert isinstance(batch, (tuple, list))
            batch = [batch, [self.fake_tensor(batch[0])]]
        if args.pipeline_type == "gpipe":
            loss = model.gpipe_forward(batch, loss_func)
            if profiler is not None:
                profiler.profile_memory(self.iter, "After Forward")
            model.gpipe_backward()
        elif args.pipeline_type == "pipedream_flush":
            loss = model.pipedream_flush_forward_backward(batch, loss_func)
        self.iter += 1
        return self.loss_to_cpu(loss)
    
    def fake_tensor(self, x):
        return torch.zeros([x.shape[0], 1], dtype=x.dtype, device=x.device)
    
    def fake_loss_func(self, labels, outputs):
        return outputs[0]
    
    def loss_to_cpu(self, loss):
        if isinstance(loss, (list, tuple)): # Average loss of each microbatch
            if len(loss) == 0:
                return None
            loss = np.mean([l.item() for l in loss])
        else:
            loss = loss.item()
        return loss

class GalvatronModelWrapper():
    def __init__(self, args, wrap_block_names=[]):
        self.args = args
        self.wrap_block_names = wrap_block_names
    
    # Wrap Galvatron Hybrid Parallel Model, need to be called after Galvatron is initialized
    def wrap_model_hybrid_parallel(self, model, model_config, hybrid_parallel_configs, model_info, construct_sequential_model, construct_tensor_parallel_model):
        return construct_hybrid_parallel_model_api(
            model,
            model_config,
            self.args,
            hybrid_parallel_configs,
            model_info,
            construct_sequential_model,
            construct_tensor_parallel_model,
            self.wrap_block_names
        )

    # Wrap Data Parallel Model, can be called on any PyTorch Model even when Galvatron is not initilized
    def wrap_model_data_parallel(self, model, device, dp_type='ddp', mixed_precision='bf16', comm_group=None, initialize_on_meta=False, backward_prefetch=True):
        from galvatron.core.parallel import wrap_model_data_parallel
        mixed_precision = mixed_precision_dtype(mixed_precision)
        return wrap_model_data_parallel(model, device, self.wrap_block_names, dp_type, mixed_precision, comm_group, initialize_on_meta, backward_prefetch)

    # Wrap Activation Checkpoint Model, can be called on any PyTorch Model even when Galvatron is not initilized
    def wrap_model_checkpoint(self, model):
        from galvatron.core.parallel import wrap_model_checkpoint
        return wrap_model_checkpoint(model, self.wrap_block_names)

def construct_hybrid_parallel_model_api(
    model,
    model_config,
    training_args,
    hybrid_parallel_configs,
    model_info,
    construct_sequential_model,
    construct_tensor_parallel_model,
    wrap_block_name=None,
):
    config, args, hp_configs = model_config, training_args, hybrid_parallel_configs

    # Get model-specific model info: module_types, layernum_list, layer_shapes_list, layer_dtypes_list
    model_info = model_info(config, args)
    module_types = model_info.module_types()
    layernum_list = model_info.layernums()
    layer_shapes_list = model_info.shapes()
    layer_dtypes_list = model_info.dtypes()
    
    # Check the validity of hp_configs (encoders only)
    check_hp_config(hp_configs, layernum_list)
    
    # Calculate shapes and dtypes for whole model (including embed/cls/... layers)
    shapes_whole, dtypes_whole = layer_shapes_dtypes_whole_model(module_types, layernum_list, layer_shapes_list, layer_dtypes_list)
    
    # Get hp_configs_whole for the whole model (including embed/cls/... layers)
    hp_configs_whole = hp_config_whole_model(module_types, hp_configs, embed_sdp=args.embed_sdp, embed_ckpt=0)

    # [Step 0] Generate communication groups
    pp_group, tp_groups_whole, dp_groups_whole, allgather_groups_whole, split_groups_whole = \
        gen_comm_groups(hp_configs_whole['tp_sizes_whole'], hp_configs_whole['pp_deg'], hp_configs_whole['tp_consec_whole'], show_rank = 0)
    
    # [Step 1] Construct Tensor Parallel Model based on tp_groups using model-specific TP function
    model = construct_tensor_parallel_model(model, config, get_enc_groups(tp_groups_whole, module_types))

    # [Step 2] Construct Sequantial model using model-specific sequential function
    model = construct_sequential_model(model, config)

    # [Step 3] Wrap Relocation modules if necessary
    model = wrap_modules_relocation(model, allgather_groups_whole, split_groups_whole)

    # [Step 4] Construct Pipeline Module and place the layers on corresponding devices
    from galvatron.core.pipeline import PipelineParallel
    hp_model = PipelineParallel(
        model=model, 
        model_ranks=hp_configs_whole['pp_ranks_whole'], 
        layer_output_tensor_shapes=shapes_whole, 
        layer_output_tensor_dtypes=dtypes_whole,
        layer_dp_sizes=hp_configs_whole['dp_sizes_whole'], 
        chunks=get_chunks(args), 
        process_group=pp_group.ranks, 
        nproc_per_node=8,
        info=False
    )

    # [Step 5] Wrap Data Parallel modules based on dp_types & dp_groups
    hp_model.wrap_pipeline_modules_data_parallel(
        hp_configs_whole['dp_types_whole'], 
        dp_groups_whole, 
        module_types=module_types, 
        mixed_precision=mixed_precision_dtype(args.mixed_precision), 
        wrap_block_name=wrap_block_name
    )
    
    # [Step 6] Wrap checkpoint based on checkpoint_flags
    hp_model.wrap_pipeline_modules_checkpoint(hp_configs_whole['checkpoint_flags_whole'], wrap_block_name=wrap_block_name)
    
    model = GalvatronModel(hp_model)
    return model