import os, json
from transformers import GPT2Config
from galvatron.utils import dict_join_dirname

# ============= Meta AI Model Config Paths =============
path_dict =  {
    'gpt-0.3b': 'gpt-0.3b.json',
    'gpt-1.5b': 'gpt-1.5b.json',
    'gpt-2.7b': 'gpt-2.7b.json',
    'gpt-6.7b': 'gpt-6.7b.json',
}

def config_from_meta(model_type) -> GPT2Config:
    global path_dict
    path_dict = dict_join_dirname(path_dict, os.path.dirname(__file__))
    with open(path_dict[model_type]) as f:
        params = json.load(f)
    return GPT2Config(**params)

# ============= Set Model Config and Arguments =============
def set_model_config(config, args, overwrite_args=True):
    config.use_cache = False
    config.fused_bias_fc = True
    config.sequence_parallel = False
    config.use_flash_attn = hasattr(args, 'use_flash_attn') and args.use_flash_attn
    
    # ======= Arguments --> Model Config ======
    # Overwrite all model configs by manually set arguments
    if args.set_model_config_manually:
        config.vocab_size = args.vocab_size
        config.hidden_size = args.hidden_size
        config.num_hidden_layers = args.num_hidden_layers
        config.num_attention_heads = args.num_attention_heads
        config.max_position_embeddings = args.seq_length
        config.resid_pdrop = args.dropout_prob
        config.embd_pdrop = args.dropout_prob
        config.attn_pdrop = args.dropout_prob
    # Overwrite layer number only
    elif args.set_layernum_manually:
        config.num_hidden_layers = args.num_hidden_layers
    
    # ======= Model Config --> Arguments ======
    # This step is necessary that maintains the consistency of model config and arguments.
    # Overwrite the model arguments with the model config
    overwrite_model_args(config, args)
    
    if overwrite_args: # Overwrite necessary Megatron-LM arguments with the model config
        overwrite_megatron_args(config, args)
    return config

def overwrite_megatron_args(config, args):
    args.hidden_size = config.hidden_size
    args.num_layers = config.num_hidden_layers
    args.num_attention_heads = config.num_attention_heads
    args.ffn_hidden_size = args.hidden_size * 4
    args.max_position_embeddings = config.max_position_embeddings
    args.use_cpu_initialization = True

# Need to overwrite the arguments with the model config
def overwrite_model_args(config, args):
    args.hidden_size = config.hidden_size
    args.seq_length = config.max_position_embeddings
    args.num_hidden_layers = config.num_hidden_layers
    args.vocab_size = config.vocab_size
    args.num_attention_heads = config.num_attention_heads

# ============= Get Model Name and Layer Configs =============
def model_name(config, args=None):
    return 'hidden%d_head%d_seqlen%d'%(config.hidden_size, config.num_attention_heads, config.max_position_embeddings)

def model_layer_configs(config):
    return [
        {
            'hidden_size': config.hidden_size,
            'seq_len': config.max_position_embeddings,
            'layer_num': config.num_hidden_layers
        }
    ]