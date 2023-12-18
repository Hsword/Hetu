import os, json
from transformers import GPT2Config, LlamaConfig
from galvatron.utils import dict_join_dirname

# ============= Meta AI Model Config Paths =============
path_dict =  {
    'llama-7b': 'llama-7b.json',
    'llama-13b': 'llama-13b.json',
    'llama-30b': 'llama-30b.json',
}

def config_from_meta(model_type) -> LlamaConfig:
    global path_dict
    path_dict = dict_join_dirname(path_dict, os.path.dirname(__file__))
    with open(path_dict[model_type]) as f:
        params = json.load(f)
    return LlamaConfig(
        hidden_size=params['dim'], intermediate_size=None,
        num_attention_heads=params['n_heads'],
        num_hidden_layers=params['n_layers'],
        rms_norm_eps=params['norm_eps']
    )

def llama_config_to_gpt2_config(llama_config: LlamaConfig) -> GPT2Config:
    return GPT2Config(
        vocab_size=llama_config.vocab_size,
        n_positions=llama_config.max_position_embeddings,
        n_embd=llama_config.hidden_size,
        n_layer=llama_config.num_hidden_layers,
        n_head=llama_config.num_attention_heads,
        n_inner=llama_config.intermediate_size,
        activation_function='swiglu',  # Hardcode since HF calls it 'silu'
        # Llama doesn't have dropout, idk if it's because they only release the inference code
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        layer_norm_epsilon=llama_config.rms_norm_eps,
        initializer_range=llama_config.initializer_range,
        bos_token_id=llama_config.bos_token_id,
        eos_token_id=llama_config.eos_token_id,
        # These are new arguments not in the original GPT2Config
        pad_token_id=llama_config.pad_token_id,  # Idk if this does anything
        rms_norm=True,
        rotary_emb_fraction=1.0,
        rotary_emb_interleaved=True,
        tie_word_embeddings=False,
        qkv_proj_bias=False,
        out_proj_bias=False,
        mlp_fc1_bias=False,
        mlp_fc2_bias=False,
    )

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