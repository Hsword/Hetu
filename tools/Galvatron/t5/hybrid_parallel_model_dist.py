import os
import sys
sys.path.insert(0, '..')
import torch
from torch import nn
import numpy as np
from utils import gen_groups_dist, modules_to_devices, wrap_modules_data_parallel, wrap_modules_relocation, wrap_data_parallel
from utils import read_json_config, config2strategy, str2array
from pipeline import PipelineParallel, PipeSequential

def overwrite_megatron_args(config, args):
    args.hidden_size = config.d_model
    args.num_layers = config.num_layers + config.num_decoder_layers
    args.num_attention_heads = config.num_heads
    args.ffn_hidden_size = config.d_ff
    args.max_position_embeddings = config.n_positions
    args.attention_dropout = config.dropout_rate
    args.hidden_dropout = config.dropout_rate
    args.use_cpu_initialization = True

def get_chunks(args):
    if args.chunks == -1:
        args.chunks = 1
        if args.pp_deg > 1:
            world_size = torch.distributed.get_world_size()
            max_dp_deg = world_size // args.pp_deg
            local_bsz = args.global_train_batch_size // max_dp_deg
            optimal_micro_bsz = np.ceil(local_bsz / 4)
            optimal_micro_bsz = 1 if optimal_micro_bsz == 0 else optimal_micro_bsz
            args.chunks = int(optimal_micro_bsz)
    return args.chunks

### Memory-balanced pipeline partition example
# T5 Large 48 pp divide
pp_stage_dict_for_bsz_48 = {8: {1: [48], 2: [29, 19], 4: [14, 15, 11, 8], 8: [8, 9, 8, 5, 5, 5, 5, 3], 16: [1, 2, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 1]}, 
                        16: {1: [48], 2: [29, 19], 4: [17, 13, 10, 8], 8: [8, 9, 8, 5, 5, 5, 5, 3], 16: [5, 5, 5, 5, 4, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 1]}, 
                        24: {1: [48], 2: [29, 19], 4: [17, 13, 10, 8], 8: [8, 8, 8, 5, 5, 5, 5, 4], 16: [5, 5, 5, 5, 4, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 1]}, 
                        32: {1: [48], 2: [30, 18], 4: [17, 13, 10, 8], 8: [8, 8, 8, 5, 5, 5, 5, 4], 16: [5, 5, 5, 5, 4, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 1]}, 
                        40: {1: [48], 2: [29, 19], 4: [15, 14, 10, 9], 8: [8, 8, 8, 5, 5, 5, 5, 4], 16: [5, 5, 5, 5, 4, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 1]}, 
                        48: {1: [48], 2: [29, 19], 4: [15, 14, 10, 9], 8: [8, 8, 8, 5, 5, 5, 5, 4], 16: [5, 5, 5, 5, 4, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 1]}, 
                        56: {1: [48], 2: [29, 19], 4: [15, 14, 10, 9], 8: [8, 8, 8, 5, 5, 5, 5, 4], 16: [5, 5, 5, 5, 4, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2]}, 
                        64: {1: [48], 2: [29, 19], 4: [15, 14, 10, 9], 8: [8, 8, 8, 5, 5, 5, 5, 4], 16: [5, 5, 5, 5, 4, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2]}, 
                        72: {1: [48], 2: [29, 19], 4: [16, 13, 10, 9], 8: [8, 8, 8, 5, 5, 5, 5, 4], 16: [5, 5, 5, 5, 4, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2]}, 
                        80: {1: [48], 2: [29, 19], 4: [16, 13, 10, 9], 8: [8, 8, 8, 5, 5, 5, 5, 4], 16: [5, 5, 5, 5, 4, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2]}, 
                        88: {1: [48], 2: [29, 19], 4: [16, 13, 10, 9], 8: [8, 8, 8, 5, 5, 5, 5, 4], 16: [5, 5, 5, 5, 4, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2]}, 
                        96: {1: [48], 2: [29, 19], 4: [16, 13, 10, 9], 8: [8, 8, 8, 5, 5, 5, 5, 4], 16: [5, 5, 5, 5, 4, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2]}, 
                        104: {1: [48], 2: [29, 19], 4: [16, 13, 10, 9], 8: [8, 8, 8, 5, 5, 5, 5, 4], 16: [5, 5, 5, 5, 4, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2]}, 
                        112: {1: [48], 2: [29, 19], 4: [16, 13, 10, 9], 8: [8, 8, 8, 5, 5, 5, 5, 4], 16: [5, 5, 5, 5, 4, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2]}, 
                        120: {1: [48], 2: [29, 19], 4: [16, 13, 10, 9], 8: [8, 8, 8, 5, 5, 5, 5, 4], 16: [5, 5, 5, 5, 4, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2]}}

# T5 Large 32 pp divide
pp_stage_dict_for_bsz_32 = {8: {1: [32], 2: [20, 12], 4: [8, 11, 8, 5], 8: [3, 6, 6, 4, 4, 4, 4, 1], 16: [1, 1, 2, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 1, 1]}, 
                        16: {1: [32], 2: [20, 12], 4: [11, 9, 7, 5], 8: [7, 7, 4, 3, 3, 3, 3, 2], 16: [1, 1, 1, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1]}, 
                        24: {1: [32], 2: [20, 12], 4: [11, 9, 7, 5], 8: [7, 7, 4, 3, 3, 3, 3, 2], 16: [1, 1, 1, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1]}, 
                        32: {1: [32], 2: [20, 12], 4: [11, 9, 7, 5], 8: [7, 7, 4, 3, 3, 3, 3, 2], 16: [1, 1, 1, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1]}, 
                        40: {1: [32], 2: [20, 12], 4: [11, 9, 7, 5], 8: [7, 7, 4, 3, 3, 3, 3, 2], 16: [1, 1, 1, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1]}, 
                        48: {1: [32], 2: [20, 12], 4: [11, 9, 7, 5], 8: [7, 7, 4, 3, 3, 3, 3, 2], 16: [1, 1, 1, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1]}, 
                        56: {1: [32], 2: [20, 12], 4: [9, 10, 7, 6], 8: [7, 7, 4, 3, 3, 3, 3, 2], 16: [1, 1, 1, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1]}, 
                        64: {1: [32], 2: [20, 12], 4: [9, 10, 7, 6], 8: [7, 7, 4, 3, 3, 3, 3, 2], 16: [1, 1, 1, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1]}, 
                        72: {1: [32], 2: [20, 12], 4: [10, 9, 7, 6], 8: [7, 7, 4, 3, 3, 3, 3, 2], 16: [1, 1, 1, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1]}, 
                        80: {1: [32], 2: [20, 12], 4: [10, 9, 7, 6], 8: [7, 7, 4, 3, 3, 3, 3, 2], 16: [1, 1, 1, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1]}, 
                        88: {1: [32], 2: [20, 12], 4: [10, 9, 7, 6], 8: [7, 7, 4, 3, 3, 3, 3, 2], 16: [1, 1, 1, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1]}, 
                        96: {1: [32], 2: [20, 12], 4: [10, 9, 7, 6], 8: [7, 7, 4, 3, 3, 3, 3, 2], 16: [1, 1, 1, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1]}, 
                        104: {1: [32], 2: [20, 12], 4: [10, 9, 7, 6], 8: [7, 7, 4, 3, 3, 3, 3, 2], 16: [1, 1, 1, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1]}, 
                        112: {1: [32], 2: [20, 12], 4: [10, 9, 7, 6], 8: [7, 7, 4, 3, 3, 3, 3, 2], 16: [1, 1, 1, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1]}, 
                        120: {1: [32], 2: [20, 12], 4: [10, 9, 7, 6], 8: [7, 7, 4, 3, 3, 3, 3, 2], 16: [1, 1, 1, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1]}}

def get_pp_ranks_enc(pp_divide):
    pp_ranks_enc = []
    pp_deg = len(pp_divide)
    for i in range(pp_deg):
        pp_ranks_enc += [i]*pp_divide[i]
    return pp_ranks_enc

def get_hybrid_parallel_configs(args):
    local_rank = args.local_rank
    world_size = torch.distributed.get_world_size()
    total_layer_num = args.num_encoder_layer+args.num_decoder_layer
    config_type = 'JSON' if args.galvatron_config_path not in [None,'None'] else 'PYTHON' if args.apply_strategy else 'GLOBAL'
    if local_rank == 0:
        print('======================== Galvatron Parallel Config =============================')
        print('Galvatron parallel config mode: [%s config mode]'%config_type)
    if config_type == 'GLOBAL':
        pp_deg = args.pp_deg
        tp_sizes_enc = [args.global_tp_deg] * total_layer_num if args.global_tp_deg > 0 else [1]*total_layer_num
        tp_consecutive_flags = [args.global_tp_consec] * total_layer_num if args.global_tp_consec in [0, 1] else [1]*total_layer_num
        dp_types_enc = total_layer_num * [args.fsdp]
    else:
        if config_type == 'JSON':
            galvatron_config = read_json_config(args.galvatron_config_path)
            pp_deg, tp_sizes_enc, tp_consecutive_flags, dp_types_enc = config2strategy(galvatron_config)
            bsz, chunks = galvatron_config['global_bsz'], galvatron_config['chunks']
            pp_divide = str2array(galvatron_config['pp_division'])
            config_source = 'Galvatron JSON config %s'%args.galvatron_config_path
        elif config_type == 'PYTHON':
            tp_sizes_enc, tp_consecutive_flags, dp_types_enc, pp_deg, bsz, chunks, pp_divide = apply_strategy()
            config_source = 'Galvatron PYTHON config'
        pp_ranks_enc = get_pp_ranks_enc(pp_divide)
        if local_rank == 0 and (total_layer_num != len(tp_sizes_enc) or args.chunks != chunks or args.global_train_batch_size != bsz):
            print('[Notice] The following hyper-parameters will be overwritten by Galvatron %s config:'%config_type)
            if args.global_train_batch_size != bsz:
                print('   global_batch_size =', bsz)
            if args.chunks != chunks:
                print('   chunks =', chunks)
            if total_layer_num != len(tp_sizes_enc):
                print('   num_hidden_layers =', len(tp_sizes_enc))
        args.global_train_batch_size = bsz
        args.chunks = chunks
        args.num_encoder_layer, args.num_decoder_layer = len(tp_sizes_enc) // 2, len(tp_sizes_enc) // 2

    if config_type == 'GLOBAL':
        if total_layer_num == 32:
            pp_divide = pp_stage_dict_for_bsz_32[min(args.global_train_batch_size,120)][pp_deg]
        elif total_layer_num == 48:
            pp_divide = pp_stage_dict_for_bsz_48[min(args.global_train_batch_size,120)][pp_deg]
        else:
            avg_layer_num = int(total_layer_num // pp_deg)
            last_layer_num = total_layer_num - avg_layer_num * (pp_deg-1)
            pp_divide = [avg_layer_num] * (pp_deg-1) + [last_layer_num]
        pp_ranks_enc = get_pp_ranks_enc(pp_divide)

    assert args.global_train_batch_size % (world_size//pp_deg) == 0, 'global_train_batch_size should be multiple of world_size//pp_deg!'
    if local_rank == 0:
        if config_type == 'GLOBAL':
            print('[GLOBAL config mode] Loaded global hybrid parallel strategy:')
            dp_type = 'sdp' if args.fsdp else 'dp'
            tp_deg, tp_consec = tp_sizes_enc[0], tp_consecutive_flags[0]
            dp_deg = world_size//args.global_tp_deg//args.pp_deg
            print('   global_batch_size: %d, chunks: %d'%(args.global_train_batch_size, get_chunks(args)))
            print('   pp_deg: %d, tp_deg: %d, %s_deg: %d, tp_consecutive_flag: %d'%(pp_deg, tp_deg, dp_type, dp_deg, tp_consec))
            print('   pp_division:\t', pp_divide)
            print('   pp_ranks:\t', pp_ranks_enc)
        else:
            print('[%s config mode] Loaded hybrid parallel config from %s:'%(config_type, config_source))
            print('   global_batch_size: %d, chunks: %d, pp_deg: %d'%(args.global_train_batch_size, args.chunks, pp_deg))
            print('   tp_sizes_enc:\t', tp_sizes_enc)
            print('   tp_consecutive_flags:', tp_consecutive_flags)
            print('   dp_types_enc:\t', dp_types_enc)
            print('   pp_division:\t\t', pp_divide)
            print('   pp_ranks:\t\t', pp_ranks_enc)
        print('================================================================================')
    hybrid_parallel_configs =   {'pp_deg':pp_deg,
                                'tp_sizes_enc':tp_sizes_enc,
                                'tp_consecutive_flags':tp_consecutive_flags,
                                'dp_types_enc':dp_types_enc,
                                'pp_ranks_enc':pp_ranks_enc}
    return hybrid_parallel_configs

def construct_hybrid_parallel_model(model, model_config, training_args, hybrid_parallel_configs):
    t5_model, config, args, hp_configs = model, model_config, training_args, hybrid_parallel_configs
    pp_deg, tp_sizes_enc, tp_consecutive_flags, dp_types_enc, pp_ranks_enc = \
        hp_configs['pp_deg'], hp_configs['tp_sizes_enc'], hp_configs['tp_consecutive_flags'], hp_configs['dp_types_enc'], hp_configs['pp_ranks_enc']
    assert config.num_layers + config.num_decoder_layers == len(tp_sizes_enc)
    assert config.num_layers + config.num_decoder_layers == len(dp_types_enc) 
    assert config.num_layers + config.num_decoder_layers == len(pp_ranks_enc)
    world_size = torch.distributed.get_world_size()
    for tp_size in tp_sizes_enc:
        assert tp_size <= world_size//pp_deg and (world_size//pp_deg) % tp_size == 0 and tp_size >= 1, 'Wrong tp_size!'
    for dp_type in dp_types_enc:
        assert dp_type == 0 or dp_type == 1 or dp_type is None, 'Wrong dp_type!'
    for pp_rank in pp_ranks_enc:
        assert pp_rank >= 0 and pp_rank <= pp_deg - 1, 'Wrong pp_rank!'

    # [Step 0] Construct sizes & groups
    # Construct tp_sizes / dp_types / pp_stages for whole model
    tp_sizes_whole_model  = [1] + tp_sizes_enc[:config.num_layers] + [1] + tp_sizes_enc[config.num_layers:] + [1]
    dp_types_whole_model  = [0] + dp_types_enc[:config.num_layers] + [0] + dp_types_enc[config.num_layers:] + [0]
    pp_ranks_whole_model = [0] + pp_ranks_enc[:config.num_layers] + [pp_ranks_enc[config.num_layers]] + pp_ranks_enc[config.num_layers:] + [pp_deg-1]
    tp_consecutive_whole_model = [1] + tp_consecutive_flags[:config.num_layers] + [1] + tp_consecutive_flags[config.num_layers:] + [1]
    # Construct tp_groups / dp_groups / allgather_groups / slice_funcs
    pp_group, tp_groups_whole_model, dp_groups_whole_model, allgather_groups_whole_model, slice_funcs_whole_model = gen_groups_dist(tp_sizes_whole_model, pp_deg, tp_consecutive_whole_model, show_rank = 0)
    tp_groups_enc = tp_groups_whole_model[1:-2]

    # [Step 1] Construct Tensor Parallel Block based on tp_groups
    from T5ForConditionalGeneration_tensor_parallel import T5LayerFF_tp, T5Attention_tp, T5Block_tp
    from T5ForConditionalGeneration_tensor_parallel import get_extended_attention_mask_encoder, get_extended_attention_mask_decoder, invert_attention_mask

    self_config = t5_model.encoder.config
    for i in range(config.num_layers):
        layer = t5_model.encoder.block[i].layer
        setattr(layer[0], 'SelfAttention', T5Attention_tp(self_config, tp_group=tp_groups_enc[i]))
        layer[-1] = T5LayerFF_tp(self_config, tp_group=tp_groups_enc[i])
        setattr(t5_model.encoder.block[i], 'layer', layer)
        t5_model.encoder.block[i] = T5Block_tp(t5_model.encoder.block[i])
    setattr(t5_model.encoder, 'get_extended_attention_mask', get_extended_attention_mask_encoder)

    cross_config = t5_model.decoder.config
    for i in range(config.num_decoder_layers):
        layer = t5_model.decoder.block[i].layer
        setattr(layer[0], 'SelfAttention', T5Attention_tp(self_config, tp_group=tp_groups_enc[i+config.num_layers]))
        setattr(layer[1], 'EncDecAttention', T5Attention_tp(cross_config, tp_group=tp_groups_enc[i+config.num_layers]))
        layer[-1] = T5LayerFF_tp(cross_config, tp_group=tp_groups_enc[i+config.num_layers])
        setattr(t5_model.decoder.block[i], 'layer', layer)
        t5_model.decoder.block[i] = T5Block_tp(t5_model.decoder.block[i])
    setattr(t5_model.decoder, 'get_extended_attention_mask', get_extended_attention_mask_decoder)
    setattr(t5_model.decoder, 'invert_attention_mask', invert_attention_mask)

    # [Step 2] Construct Sequantial modules
    from T5ForConditionalGeneration_pipeline import T5Embeddings_, T5Decoder_, T5Cls_, T5Encoder_, T5DecoderEmbedding_
    model = PipeSequential()
    model.add_module('embeddings_1', T5Embeddings_(t5_model))
    for i in range(config.num_layers):
        model.add_module('encoder_%d'%(i), 
            T5Encoder_(t5_model, i, has_final_layernorm= i + 1 >= config.num_layers))
            # T5Encoder_(t5_model, i, i + 1, has_final_layernorm= i + 1 >= config.num_layers))

    model.add_module('embeddings_2', T5DecoderEmbedding_(t5_model))
    for i in range(config.num_decoder_layers):
        model.add_module('decoder_%d'%(i), 
            T5Decoder_(t5_model, i, has_final_layernorm= i + 1 >= config.num_decoder_layers))
            # T5Decoder_(t5_model, i, i + 1, has_final_layernorm= i + 1 >= config.num_decoder_layers))
    model.add_module('cls', T5Cls_(t5_model))

    # [Step 3] Wrap Relocation modules if necessary
    model = wrap_modules_relocation(model, allgather_groups_whole_model, slice_funcs_whole_model)

    # [Step 4] Place Sequantial modules to GPU devices based on pp_stages
    chunks = get_chunks(args)
    seq_len, hidden_size = args.seq_length, config.d_model
    layer_output_tensor_shapes = [None] + [[[-1,seq_len,hidden_size], [-1,seq_len], [-1,seq_len]]] * config.num_layers \
                                 + [[[-1,seq_len,hidden_size], [-1,seq_len], [-1,seq_len,hidden_size], [-1,seq_len]]] * (1+config.num_decoder_layers) + [None]
    layer_output_tensor_dtypes = [None] + [[torch.float, torch.long, torch.long]] * config.num_layers \
                                 + [[torch.float, torch.long, torch.float, torch.long]] * (1+config.num_decoder_layers) + [None]
    layer_dp_sizes = [world_size // pp_deg // tp_size for tp_size in tp_sizes_whole_model]
    hp_model = PipelineParallel(
                model = model, 
                model_ranks = pp_ranks_whole_model, 
                layer_output_tensor_shapes = layer_output_tensor_shapes, 
                layer_output_tensor_dtypes = layer_output_tensor_dtypes,
                layer_dp_sizes = layer_dp_sizes, 
                chunks=args.chunks, 
                process_group = pp_group.ranks, 
                nproc_per_node=8,
                info=False)

    # [Step 5] Wrap Data Parallel modules based on dp_types & dp_groups
    module_types = ['embed_1'] + ['t5_enc']*config.num_layers + ['embed_2'] + ['t5_dec']*config.num_decoder_layers + ['cls']
    hp_model.wrap_pipeline_modules_data_parallel(dp_types_whole_model, dp_groups_whole_model, module_types=module_types)
    return hp_model


def apply_strategy():
    ### Example strategy
    pp_deg = 2
    tp_sizes_enc = [2]*12+[1]*36
    tp_consecutive_flags = [1]*48
    dp_types_enc = [0]*36+[1]*12
    global_bsz = 16
    chunks = 2
    pp_division = pp_stage_dict_for_bsz_48[global_bsz][pp_deg] # [29, 19]

    return tp_sizes_enc, tp_consecutive_flags, dp_types_enc, pp_deg, global_bsz, chunks, pp_division