import os
import sys
sys.path.insert(0, '..')
import torch
from torch import nn
import numpy as np
from utils import gen_groups, show_groups, modules_to_devices, wrap_modules_data_parallel, wrap_modules_relocation, wrap_data_parallel
from utils import read_json_config, config2strategy, str2array
from torch.distributed.pipeline.sync import Pipe

def get_chunks(args):
    if args.chunks == -1:
        args.chunks = 1
        if args.global_tp_deg > 0 and args.pp_deg > 1:
            world_size = torch.distributed.get_world_size()
            global_dp_deg = world_size // args.global_tp_deg
            bsz_per_gpu = args.global_train_batch_size/global_dp_deg 
            if bsz_per_gpu <= 8:
                optimal_micro_bsz = 1
            elif bsz_per_gpu > 8 and bsz_per_gpu < 32:
                optimal_micro_bsz = 2
            elif bsz_per_gpu >= 32 and bsz_per_gpu <= 96:
                optimal_micro_bsz = 3
            else:
                optimal_micro_bsz = 4
            args.chunks = int(np.ceil(optimal_micro_bsz))
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
pp_stage_dict_for_bsz_32 = {8: {1: [32], 2: [12, 20], 4: [5, 10, 10, 7], 8: [1, 4, 5, 5, 5, 5, 5, 2]}, 
                        16: {1: [32], 2: [12, 20], 4: [4, 10, 10, 8], 8: [1, 4, 5, 5, 5, 5, 5, 2]}, 
                        24: {1: [32], 2: [11, 21], 4: [4, 10, 10, 8], 8: [1, 3, 5, 5, 5, 5, 5, 3]}, 
                        32: {1: [32], 2: [11, 21], 4: [3, 10, 10, 9], 8: [1, 2, 5, 5, 5, 5, 5, 4]}, 
                        40: {1: [32], 2: [11, 21], 4: [3, 10, 10, 9], 8: [1, 2, 5, 5, 5, 5, 5, 4]}, 
                        48: {1: [32], 2: [11, 21], 4: [3, 10, 10, 9], 8: [1, 2, 5, 5, 5, 5, 5, 4]}, 
                        56: {1: [32], 2: [11, 21], 4: [3, 10, 10, 9], 8: [1, 2, 5, 5, 5, 5, 5, 4]}, 
                        64: {1: [32], 2: [11, 21], 4: [3, 9, 10, 10], 8: [1, 2, 5, 5, 5, 5, 5, 4]}, 
                        72: {1: [32], 2: [11, 21], 4: [3, 9, 10, 10], 8: [1, 2, 4, 5, 5, 5, 5, 5]}, 
                        80: {1: [32], 2: [11, 21], 4: [3, 9, 10, 10], 8: [1, 2, 4, 5, 5, 5, 5, 5]}, 
                        88: {1: [32], 2: [11, 21], 4: [3, 9, 10, 10], 8: [1, 2, 4, 5, 5, 5, 5, 5]}, 
                        96: {1: [32], 2: [11, 21], 4: [3, 9, 10, 10], 8: [1, 2, 4, 5, 5, 5, 5, 5]}, 
                        104: {1: [32], 2: [11, 21], 4: [3, 9, 10, 10], 8: [1, 2, 4, 5, 5, 5, 5, 5]}, 
                        112: {1: [32], 2: [11, 21], 4: [3, 9, 10, 10], 8: [1, 2, 4, 5, 5, 5, 5, 5]}, 
                        120: {1: [32], 2: [11, 21], 4: [3, 9, 10, 10], 8: [1, 2, 4, 5, 5, 5, 5, 5]}, 
                        128: {1: [32], 2: [11, 21], 4: [3, 9, 10, 10], 8: [1, 2, 4, 5, 5, 5, 5, 5]}, 
                        136: {1: [32], 2: [11, 21], 4: [3, 9, 10, 10], 8: [1, 2, 4, 5, 5, 5, 5, 5]}, 
                        144: {1: [32], 2: [11, 21], 4: [3, 9, 10, 10], 8: [1, 2, 4, 5, 5, 5, 5, 5]}, 
                        152: {1: [32], 2: [11, 21], 4: [3, 9, 10, 10], 8: [1, 2, 4, 5, 5, 5, 5, 5]}, 
                        160: {1: [32], 2: [11, 21], 4: [3, 9, 10, 10], 8: [1, 2, 4, 5, 5, 5, 5, 5]}, 
                        168: {1: [32], 2: [11, 21], 4: [3, 9, 10, 10], 8: [1, 2, 4, 5, 5, 5, 5, 5]}, 
                        176: {1: [32], 2: [11, 21], 4: [3, 9, 10, 10], 8: [1, 2, 4, 5, 5, 5, 5, 5]}, 
                        184: {1: [32], 2: [11, 21], 4: [3, 9, 10, 10], 8: [1, 2, 4, 5, 5, 5, 5, 5]}, 
                        192: {1: [32], 2: [11, 21], 4: [3, 9, 10, 10], 8: [1, 2, 4, 5, 5, 5, 5, 5]}, 
                        200: {1: [32], 2: [11, 21], 4: [3, 9, 10, 10], 8: [1, 2, 4, 5, 5, 5, 5, 5]}}

# Swin Huge 48 pp divide
pp_stage_dict_for_bsz_48 = {8: {1: [48], 2: [20, 28], 4: [9, 14, 14, 11], 8: [3, 7, 7, 7, 7, 7, 7, 3]}, 
                        16: {1: [48], 2: [20, 28], 4: [8, 14, 14, 12], 8: [2, 7, 7, 7, 7, 7, 7, 4]},
                         24: {1: [48], 2: [19, 29], 4: [8, 14, 14, 12], 8: [2, 6, 7, 7, 7, 7, 7, 5]}, 
                         32: {1: [48], 2: [19, 29], 4: [7, 14, 14, 13], 8: [2, 5, 7, 7, 7, 7, 7, 6]}, 
                         40: {1: [48], 2: [19, 29], 4: [7, 14, 14, 13], 8: [2, 5, 7, 7, 7, 7, 7, 6]}, 
                         48: {1: [48], 2: [19, 29], 4: [7, 14, 14, 13], 8: [2, 5, 7, 7, 7, 7, 7, 6]}, 
                         56: {1: [48], 2: [19, 29], 4: [7, 14, 14, 13], 8: [2, 5, 7, 7, 7, 7, 7, 6]}, 
                         64: {1: [48], 2: [19, 29], 4: [6, 14, 14, 14], 8: [2, 5, 7, 7, 7, 7, 7, 6]}, 
                         72: {1: [48], 2: [19, 29], 4: [6, 14, 14, 14], 8: [2, 4, 7, 7, 7, 7, 7, 7]}, 
                         80: {1: [48], 2: [19, 29], 4: [6, 14, 14, 14], 8: [2, 4, 7, 7, 7, 7, 7, 7]}, 
                         88: {1: [48], 2: [19, 29], 4: [6, 14, 14, 14], 8: [2, 4, 7, 7, 7, 7, 7, 7]}, 
                         96: {1: [48], 2: [19, 29], 4: [6, 14, 14, 14], 8: [2, 4, 7, 7, 7, 7, 7, 7]}, 
                         104: {1: [48], 2: [19, 29], 4: [6, 14, 14, 14], 8: [2, 4, 7, 7, 7, 7, 7, 7]}, 
                         112: {1: [48], 2: [19, 29], 4: [6, 14, 14, 14], 8: [2, 4, 7, 7, 7, 7, 7, 7]}, 
                         120: {1: [48], 2: [19, 29], 4: [6, 14, 14, 14], 8: [2, 4, 7, 7, 7, 7, 7, 7]}, 
                         128: {1: [48], 2: [19, 29], 4: [6, 14, 14, 14], 8: [2, 4, 7, 7, 7, 7, 7, 7]}, 
                         136: {1: [48], 2: [19, 29], 4: [6, 14, 14, 14], 8: [2, 4, 7, 7, 7, 7, 7, 7]}, 
                         144: {1: [48], 2: [19, 29], 4: [6, 14, 14, 14], 8: [2, 4, 7, 7, 7, 7, 7, 7]}}

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
            pp_divide = pp_stage_dict_for_bsz_32[args.global_train_batch_size][pp_deg]
        elif args.num_layers == 48:
            pp_divide = pp_stage_dict_for_bsz_48[args.global_train_batch_size][pp_deg]
        else:
            assert(False, 'pipeline stage division should be manually set!')
        pp_ranks_enc = get_pp_ranks_enc(pp_divide)

    assert pp_deg * world_size <= 8, '[Error] pp_deg * world_size should <= 8, please set nproc_per_node as gpu_num//pp_deg!'
    assert args.global_train_batch_size % world_size == 0, 'global_train_batch_size should be multiple of world_size!'
    if local_rank == 0:
        if config_type == 'GLOBAL':
            print('[GLOBAL config mode] Loaded global hybrid parallel strategy:')
            dp_type = 'sdp' if args.fsdp else 'dp'
            tp_deg, tp_consec = tp_sizes_enc[0], tp_consecutive_flags[0]
            dp_deg = world_size//tp_deg
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
        assert tp_size <= world_size and world_size % tp_size == 0 and tp_size >= 1, 'Wrong tp_size!'
    for dp_type in dp_types_enc:
        assert dp_type == 0 or dp_type == 1 or dp_type is None, 'Wrong dp_type!'
    for pp_rank in pp_ranks_enc:
        assert pp_rank >= 0 and pp_rank <= pp_deg - 1, 'Wrong pp_rank!'
    
    # [Step 0] Construct sizes & groups
    # Construct tp_sizes / dp_types / pp_stages for whole model
    tp_sizes_whole_model  = [1] + tp_sizes_enc + [1, 1]
    dp_types_whole_model  = [0] + dp_types_enc + [0, 0]
    pp_ranks_whole_model = [0] + pp_ranks_enc + [0, 0]
    tp_consecutive_whole_model = [1] + tp_consecutive_flags + [1, 1]
    # Construct tp_groups / dp_groups / allgather_groups / slice_funcs
    tp_groups_whole_model, dp_groups_whole_model, allgather_groups_whole_model, slice_funcs_whole_model = gen_groups(tp_sizes_whole_model, tp_consecutive_whole_model, show_rank=0)
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
    model = nn.Sequential()
    model.add_module('embeddings', SwinEmbeddings_(swin_model))

    for i, d in enumerate(args.depths):
        for j in range(d):
            model.add_module('encoder_%d_%d'%(i, j), SwinBlock_(swin_model, i, j, j==d-1))
        
    model.add_module('layernorm', SwinLayernorm_(swin_model))
    model.add_module('cls', SwinCls_(swin_model))

    # [Step 3] Wrap Relocation modules if necessary
    model = wrap_modules_relocation(model, allgather_groups_whole_model, slice_funcs_whole_model)

    # [Step 4] Place Sequantial modules to GPU devices based on pp_stages
    rank = torch.distributed.get_rank()
    devices = [i * world_size + rank for i in range(pp_deg)]
    pp_devices_whole_model = [devices[i] for i in pp_ranks_whole_model]
    modules_to_devices(model, pp_devices_whole_model)

    module_types = ['embed'] + ['swin_enc']*num_hidden_layers + ['pooler', 'cls']
    # [Step 5] Wrap Data Parallel modules based on dp_types & dp_groups
    model = wrap_modules_data_parallel(model, dp_types_whole_model, dp_groups_whole_model, module_types, pp_devices=pp_devices_whole_model)

    # [Step 6] Construct Pipeline Parallel model
    chunks = get_chunks(args)
    hp_model = Pipe(model, chunks=chunks, checkpoint='never')
    return hp_model

def apply_strategy():
    ### Example strategy
    pp_deg = 2
    tp_sizes_enc = [1]*44+[2]*2+[4]*2
    tp_consecutive_flags = [1]*48
    dp_types_enc = [0]*2+[1]*4+[0]*42
    global_bsz = 64
    chunks = 2
    pp_division = pp_stage_dict_for_bsz_48[global_bsz][pp_deg] # [19, 29]

    return tp_sizes_enc, tp_consecutive_flags, dp_types_enc, pp_deg, global_bsz, chunks, pp_division