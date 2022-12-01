import os
import sys
sys.path.insert(0, '..')
import torch
from torch import nn
import numpy as np
from utils import gen_groups_dist, modules_to_devices, wrap_modules_data_parallel, wrap_modules_relocation, wrap_data_parallel
from utils import read_json_config, config2strategy, str2array
from pipeline import PipelineParallel, PipeSequential

def get_chunks(args):
    if args.chunks == -1:
        args.chunks = 1
        if args.pp_deg > 1:
            world_size = torch.distributed.get_world_size()
            max_dp_deg = world_size // args.pp_deg
            local_bsz = args.global_train_batch_size // max_dp_deg
            if args.pp_deg <= 8:
                optimal_micro_bsz = np.ceil(local_bsz / 16)
            else:
                optimal_micro_bsz = np.ceil(local_bsz / 32)
            optimal_micro_bsz = 1 if optimal_micro_bsz == 0 else optimal_micro_bsz
            args.chunks = int(optimal_micro_bsz)
    return args.chunks

def overwrite_megatron_args(config, args):
    args.num_layers = sum(config.depths)
    args.num_attention_heads = config.num_heads
    args.max_position_embeddings = config.embed_dim
    args.attention_dropout = config.attention_probs_dropout_prob
    args.hidden_dropout = config.hidden_dropout_prob
    args.use_cpu_initialization = True

### Memory-balanced pipeline partition example
# Swin Huge 32 pp divide
pp_stage_dict_for_bsz_32 = {8: {1: [32], 2: [17, 15], 4: [9, 9, 9, 5], 8: [1, 4, 5, 5, 5, 5, 5, 2], 16: [3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1]}, 
                        16: {1: [32], 2: [16, 16], 4: [5, 10, 10, 7], 8: [1, 4, 5, 5, 5, 5, 5, 2], 16: [1, 3, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1]}, 
                        24: {1: [32], 2: [15, 17], 4: [4, 10, 10, 8], 8: [1, 3, 5, 5, 5, 5, 5, 3], 16: [1, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1]}, 
                        32: {1: [32], 2: [14, 18], 4: [4, 10, 10, 8], 8: [1, 3, 5, 5, 5, 5, 5, 3], 16: [1, 1, 1, 3, 4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1]}, 
                        40: {1: [32], 2: [14, 18], 4: [3, 10, 10, 9], 8: [1, 2, 5, 5, 5, 5, 5, 4], 16: [1, 1, 1, 2, 4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]}, 
                        48: {1: [32], 2: [13, 19], 4: [3, 10, 10, 9], 8: [1, 2, 5, 5, 5, 5, 5, 4], 16: [1, 1, 1, 2, 4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]}, 
                        56: {1: [32], 2: [13, 19], 4: [3, 10, 10, 9], 8: [1, 2, 5, 5, 5, 5, 5, 4], 16: [1, 1, 1, 1, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]}, 
                        64: {1: [32], 2: [13, 19], 4: [3, 10, 10, 9], 8: [1, 2, 5, 5, 5, 5, 5, 4], 16: [1, 1, 1, 1, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]}, 
                        72: {1: [32], 2: [13, 19], 4: [3, 10, 10, 9], 8: [1, 2, 5, 5, 5, 5, 5, 4], 16: [1, 1, 1, 1, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]}, 
                        80: {1: [32], 2: [12, 20], 4: [3, 10, 10, 9], 8: [1, 2, 4, 5, 5, 5, 5, 5], 16: [1, 1, 1, 1, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]}, 
                        88: {1: [32], 2: [12, 20], 4: [3, 9, 10, 10], 8: [1, 2, 4, 5, 5, 5, 5, 5], 16: [1, 1, 1, 1, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]}, 
                        96: {1: [32], 2: [12, 20], 4: [3, 9, 10, 10], 8: [1, 2, 4, 5, 5, 5, 5, 5], 16: [1, 1, 1, 1, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2]}, 
                        104: {1: [32], 2: [12, 20], 4: [3, 9, 10, 10], 8: [1, 2, 4, 5, 5, 5, 5, 5], 16: [1, 1, 1, 1, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2]}, 
                        112: {1: [32], 2: [12, 20], 4: [3, 9, 10, 10], 8: [1, 2, 4, 5, 5, 5, 5, 5], 16: [1, 1, 1, 1, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2]}, 
                        120: {1: [32], 2: [12, 20], 4: [3, 9, 10, 10], 8: [1, 2, 4, 5, 5, 5, 5, 5], 16: [1, 1, 1, 1, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 3]}, 
                        128: {1: [32], 2: [12, 20], 4: [3, 9, 10, 10], 8: [1, 2, 4, 5, 5, 5, 5, 5], 16: [1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3]}, 
                        136: {1: [32], 2: [12, 20], 4: [3, 9, 10, 10], 8: [1, 2, 4, 5, 5, 5, 5, 5], 16: [1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3]}, 
                        144: {1: [32], 2: [12, 20], 4: [3, 9, 10, 10], 8: [1, 2, 4, 5, 5, 5, 5, 5], 16: [1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3]}, 
                        152: {1: [32], 2: [12, 20], 4: [3, 9, 10, 10], 8: [1, 2, 4, 5, 5, 5, 5, 5], 16: [1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3]}, 
                        160: {1: [32], 2: [12, 20], 4: [3, 9, 10, 10], 8: [1, 2, 4, 5, 5, 5, 5, 5], 16: [1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3]}, 
                        168: {1: [32], 2: [12, 20], 4: [3, 9, 10, 10], 8: [1, 2, 4, 5, 5, 5, 5, 5], 16: [1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3]}, 
                        176: {1: [32], 2: [12, 20], 4: [3, 9, 10, 10], 8: [1, 2, 4, 5, 5, 5, 5, 5], 16: [1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3]}, 
                        184: {1: [32], 2: [12, 20], 4: [3, 9, 10, 10], 8: [1, 2, 4, 5, 5, 5, 5, 5], 16: [1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3]}, 
                        192: {1: [32], 2: [12, 20], 4: [3, 9, 10, 10], 8: [1, 2, 4, 5, 5, 5, 5, 5], 16: [1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3]}, 
                        200: {1: [32], 2: [12, 20], 4: [3, 9, 10, 10], 8: [1, 2, 4, 5, 5, 5, 5, 5], 16: [1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3]}}


# # Swin Huge 48 pp divide
pp_stage_dict_for_bsz_48 = {8: {1: [48], 2: [25, 23], 4: [13, 13, 13, 9], 8: [3, 7, 7, 7, 7, 7, 7, 3], 16: [4, 6, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1]}, 
                        16: {1: [48], 2: [24, 24], 4: [9, 14, 14, 11], 8: [2, 7, 7, 7, 7, 7, 7, 4], 16: [2, 5, 5, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1]}, 
                        24: {1: [48], 2: [23, 25], 4: [8, 14, 14, 12], 8: [2, 6, 7, 7, 7, 7, 7, 5], 16: [1, 3, 5, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2]}, 
                        32: {1: [48], 2: [22, 26], 4: [8, 14, 14, 12], 8: [2, 6, 7, 7, 7, 7, 7, 5], 16: [1, 3, 5, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2]}, 
                        40: {1: [48], 2: [22, 26], 4: [7, 14, 14, 13], 8: [2, 5, 7, 7, 7, 7, 7, 6], 16: [1, 2, 5, 5, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2]}, 
                        48: {1: [48], 2: [21, 27], 4: [7, 14, 14, 13], 8: [2, 5, 7, 7, 7, 7, 7, 6], 16: [1, 2, 5, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]}, 
                        56: {1: [48], 2: [21, 27], 4: [7, 14, 14, 13], 8: [2, 5, 7, 7, 7, 7, 7, 6], 16: [1, 2, 5, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]}, 
                        64: {1: [48], 2: [21, 27], 4: [7, 14, 14, 13], 8: [2, 5, 7, 7, 7, 7, 7, 6], 16: [1, 2, 4, 5, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]}, 
                        72: {1: [48], 2: [21, 27], 4: [7, 14, 14, 13], 8: [2, 5, 7, 7, 7, 7, 7, 6], 16: [1, 2, 4, 5, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]}, 
                        80: {1: [48], 2: [20, 28], 4: [7, 14, 14, 13], 8: [2, 4, 7, 7, 7, 7, 7, 7], 16: [1, 2, 4, 5, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]}, 
                        88: {1: [48], 2: [20, 28], 4: [6, 14, 14, 14], 8: [2, 4, 7, 7, 7, 7, 7, 7], 16: [1, 2, 4, 5, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]}, 
                        96: {1: [48], 2: [20, 28], 4: [6, 14, 14, 14], 8: [2, 4, 7, 7, 7, 7, 7, 7], 16: [1, 2, 4, 5, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]}, 
                        104: {1: [48], 2: [20, 28], 4: [6, 14, 14, 14], 8: [2, 4, 7, 7, 7, 7, 7, 7], 16: [1, 2, 4, 5, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]}, 
                        112: {1: [48], 2: [20, 28], 4: [6, 14, 14, 14], 8: [2, 4, 7, 7, 7, 7, 7, 7], 16: [1, 2, 4, 5, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]}, 
                        120: {1: [48], 2: [20, 28], 4: [6, 14, 14, 14], 8: [2, 4, 7, 7, 7, 7, 7, 7], 16: [1, 2, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4]}, 
                        128: {1: [48], 2: [20, 28], 4: [6, 14, 14, 14], 8: [2, 4, 7, 7, 7, 7, 7, 7], 16: [1, 1, 1, 1, 1, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]}, 
                        136: {1: [48], 2: [20, 28], 4: [6, 14, 14, 14], 8: [2, 4, 7, 7, 7, 7, 7, 7], 16: [1, 1, 1, 1, 1, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]}, 
                        144: {1: [48], 2: [20, 28], 4: [6, 14, 14, 14], 8: [2, 4, 7, 7, 7, 7, 7, 7], 16: [1, 1, 1, 1, 1, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]}, 
                        152: {1: [48], 2: [20, 28], 4: [6, 14, 14, 14], 8: [2, 4, 7, 7, 7, 7, 7, 7], 16: [1, 1, 1, 1, 1, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]}, 
                        160: {1: [48], 2: [20, 28], 4: [6, 14, 14, 14], 8: [2, 4, 7, 7, 7, 7, 7, 7], 16: [1, 1, 1, 1, 1, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]}, 
                        168: {1: [48], 2: [20, 28], 4: [6, 14, 14, 14], 8: [2, 4, 7, 7, 7, 7, 7, 7], 16: [1, 1, 1, 1, 1, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]}, 
                        176: {1: [48], 2: [20, 28], 4: [6, 14, 14, 14], 8: [2, 4, 7, 7, 7, 7, 7, 7], 16: [1, 1, 1, 1, 1, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]}, 
                        184: {1: [48], 2: [20, 28], 4: [6, 14, 14, 14], 8: [2, 4, 7, 7, 7, 7, 7, 7], 16: [1, 1, 1, 1, 1, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]}, 
                        192: {1: [48], 2: [20, 28], 4: [6, 14, 14, 14], 8: [2, 4, 7, 7, 7, 7, 7, 7], 16: [1, 1, 1, 1, 1, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]}, 
                        200: {1: [48], 2: [20, 28], 4: [6, 14, 14, 14], 8: [2, 4, 7, 7, 7, 7, 7, 7], 16: [1, 1, 1, 1, 1, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]}}

def get_pp_ranks_enc(pp_divide):
    pp_ranks_enc = []
    pp_deg = len(pp_divide)
    for i in range(pp_deg):
        pp_ranks_enc += [i]*pp_divide[i]
    return pp_ranks_enc

def get_hybrid_parallel_configs(args):
    local_rank = args.local_rank
    world_size = torch.distributed.get_world_size()
    args.num_layers = sum(args.depths)
    config_type = 'JSON' if args.galvatron_config_path not in [None,'None'] else 'PYTHON' if args.apply_strategy else 'GLOBAL'
    if local_rank == 0:
        print('======================== Galvatron Parallel Config =============================')
        print('Galvatron parallel config mode: [%s config mode]'%config_type)
    if config_type == 'GLOBAL':
        pp_deg = args.pp_deg
        tp_sizes_enc = [args.global_tp_deg] * args.num_layers if args.global_tp_deg > 0 else [1]*args.num_layers
        tp_consecutive_flags = [args.global_tp_consec] * args.num_layers if args.global_tp_consec in [0, 1] else [1]*args.num_layers
        dp_types_enc = args.num_layers * [args.fsdp]
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
        if local_rank == 0 and (args.num_layers != len(tp_sizes_enc) or args.chunks != chunks or args.global_train_batch_size != bsz):
            print('[Notice] The following hyper-parameters will be overwritten by Galvatron %s config:'%config_type)
            if args.global_train_batch_size != bsz:
                print('   global_batch_size =', bsz)
            if args.chunks != chunks:
                print('   chunks =', chunks)
            if args.num_layers != len(tp_sizes_enc):
                print('   num_hidden_layers =', len(tp_sizes_enc))
        args.global_train_batch_size = bsz
        args.chunks = chunks
        args.depths = [2,2,len(tp_sizes_enc)-6,2]
        args.num_layers = sum(args.depths)

    if config_type == 'GLOBAL':
        if args.num_layers == 32:
            pp_divide = pp_stage_dict_for_bsz_32[min(args.global_train_batch_size,200)][pp_deg]
        elif args.num_layers == 48:
            pp_divide = pp_stage_dict_for_bsz_48[min(args.global_train_batch_size,200)][pp_deg]
        else:
            assert(False, 'pipeline stage division should be manually set!')
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
    swin_model, config, args, hp_configs = model, model_config, training_args, hybrid_parallel_configs
    pp_deg, tp_sizes_enc, tp_consecutive_flags, dp_types_enc, pp_ranks_enc = \
        hp_configs['pp_deg'], hp_configs['tp_sizes_enc'], hp_configs['tp_consecutive_flags'], hp_configs['dp_types_enc'], hp_configs['pp_ranks_enc']
    num_hidden_layers = sum(config.depths)
    assert num_hidden_layers == len(tp_sizes_enc)
    assert num_hidden_layers == len(dp_types_enc) 
    assert num_hidden_layers == len(pp_ranks_enc)
    world_size = torch.distributed.get_world_size()
    for tp_size in tp_sizes_enc:
        assert tp_size <= world_size // pp_deg and world_size % tp_size == 0 and tp_size >= 1, 'Wrong tp_size!'
    for dp_type in dp_types_enc:
        assert dp_type == 0 or dp_type == 1 or dp_type is None, 'Wrong dp_type!'
    for pp_rank in pp_ranks_enc:
        assert pp_rank >= 0 and pp_rank <= pp_deg - 1, 'Wrong pp_rank!'
    
    # [Step 0] Construct sizes & groups
    # Construct tp_sizes / dp_types / pp_stages for whole model
    tp_sizes_whole_model  = [1] + tp_sizes_enc + [1, 1]
    dp_types_whole_model  = [0] + dp_types_enc + [0, 0]
    pp_ranks_whole_model = [0] + pp_ranks_enc + [pp_deg-1, pp_deg-1]
    tp_consecutive_whole_model = [1] + tp_consecutive_flags + [1, 1]
    # Construct tp_groups / dp_groups / allgather_groups / slice_funcs
    pp_group, tp_groups_whole_model, dp_groups_whole_model, allgather_groups_whole_model, slice_funcs_whole_model = gen_groups_dist(tp_sizes_whole_model, pp_deg, tp_consecutive_whole_model, show_rank = 0)
    tp_groups_enc = tp_groups_whole_model[1:-2]

    # [Step 1] Construct Tensor Parallel Block based on tp_groups
    from SwinForImageClassification_tensor_parallel import build_swinblock_list
    gen = tp_groups_enc.__iter__()
    for i, swinlayer in enumerate(swin_model.swin.encoder.layers):
        new_layers = build_swinblock_list(swinlayer.config, swinlayer.dim, swinlayer.blocks[0].input_resolution, 
            swinlayer.config.depths[i], swinlayer.config.num_heads[i], gen)
        setattr(swinlayer, 'blocks', new_layers)
    
    # [Step 2] Construct Sequantial modules
    from SwinForImageClassification_pipeline import SwinCls_, SwinEmbeddings_, SwinLayernorm_, SwinBlock_
    model = PipeSequential()
    model.add_module('embeddings', SwinEmbeddings_(swin_model))

    for i, d in enumerate(args.depths):
        for j in range(d):
            model.add_module('encoder_%d_%d'%(i, j), SwinBlock_(swin_model, i, j, j==d-1))
    
    model.add_module('layernorm', SwinLayernorm_(swin_model))
    model.add_module('cls', SwinCls_(swin_model))

    # [Step 3] Wrap Relocation modules if necessary
    model = wrap_modules_relocation(model, allgather_groups_whole_model, slice_funcs_whole_model)

    # [Step 4] Place Sequantial modules to GPU devices based on pp_stages
    chunks = get_chunks(args)
    seq_len, hidden_size = 196, config.embed_dim
    layer_output_tensor_shapes = [None]
    for i in range(len(config.depths)):
        seq_len, hidden_size = (config.image_size // config.patch_size // (2**i)) ** 2, config.embed_dim * (2**i)
        layer_output_tensor_shapes += (config.depths[i] - 1) * [[[-1,seq_len,hidden_size]]]
        if i < len(config.depths) -1: # downsample
            layer_output_tensor_shapes += [[[-1,seq_len//4,hidden_size*2]]]
        else:
            layer_output_tensor_shapes += [[[-1,seq_len,hidden_size]]]
    layer_output_tensor_shapes += [None] * 2
    layer_dp_sizes = [world_size // pp_deg // tp_size for tp_size in tp_sizes_whole_model]
    hp_model = PipelineParallel(
                model = model, 
                model_ranks = pp_ranks_whole_model, 
                layer_output_tensor_shapes = layer_output_tensor_shapes, 
                layer_dp_sizes = layer_dp_sizes, 
                chunks=args.chunks, 
                process_group = pp_group.ranks, 
                nproc_per_node=8,
                info=False)

    # [Step 5] Wrap Data Parallel modules based on dp_types & dp_groups
    module_types = ['embed'] + ['swin_enc']*num_hidden_layers + ['pooler', 'cls']
    hp_model.wrap_pipeline_modules_data_parallel(dp_types_whole_model, dp_groups_whole_model, module_types=module_types)

    return hp_model

def apply_strategy():
    ### Example strategy
    pp_deg = 2
    tp_sizes_enc = [1]*44+[2]*2+[4]*2
    tp_consecutive_flags = [1]*48
    dp_types_enc = [0]*2+[1]*4+[0]*42
    global_bsz = 64
    chunks = 2
    pp_division = pp_stage_dict_for_bsz_48[global_bsz][pp_deg] # [21, 27]

    return tp_sizes_enc, tp_consecutive_flags, dp_types_enc, pp_deg, global_bsz, chunks, pp_division