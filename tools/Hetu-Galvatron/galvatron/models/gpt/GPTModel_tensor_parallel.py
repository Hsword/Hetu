from galvatron.core import get_args
from flash_attn.models.gpt import create_mixer_cls, create_mlp_cls
    
def construct_tensor_parallel_model(model, config, tp_groups_enc):
    args=get_args()
    factory_kwargs = {
        'device': 'meta' if hasattr(args, 'initialize_on_meta') and args.initialize_on_meta else 'cpu',
        'dtype': None
    }
    for i in range(config.num_hidden_layers):
        layer = model.transformer.layers[i]
        setattr(layer, 'mixer', create_mixer_cls(config, layer_idx=i, process_group=tp_groups_enc[i].group, **factory_kwargs)(config.hidden_size))
        setattr(layer, 'mlp', create_mlp_cls(config, layer_idx=i, process_group=tp_groups_enc[i].group, **factory_kwargs)(config.hidden_size))
    return model