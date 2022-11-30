import sys
sys.path.insert(0, '..')
from utils import MemoryCostModel, TimeCostModel_with_overlap
from utils import DpOnModel, print_strategies, form_strategy
import numpy as np
from utils import strategy2config, read_json_config, write_json_config, read_allreduce_bandwidth_config, estimate_bsz_start_8gpus
import argparse

def read_profiling_configs(gpu_num):
    allreduce_bandwidth_config_path = '../env_configs/allreduce_bandwidth_%d_gpus.json'%gpu_num
    comm_coe_dict = read_allreduce_bandwidth_config(allreduce_bandwidth_config_path, gpu_num=gpu_num)
    overlap_coe_path = '../env_configs/overlap_coefficient.json'
    overlap_coe = read_json_config(overlap_coe_path)['overlap_coe']
    fwd_profiling_path = './configs/forward_profiling_config.json'
    fwd_time = read_json_config(fwd_profiling_path)
    return comm_coe_dict, overlap_coe, fwd_time

def parallelism_optimization(args):
    if args.type == 'full':
        # full strategies
        strategies = [[1,1,8,{'fsdp':0}],[1,1,8,{'fsdp':1}],
                    [1,2,4,{'tp':0,'fsdp':0}],[1,2,4,{'tp':1,'fsdp':0}],[1,2,4,{'tp':0,'fsdp':1}],[1,2,4,{'tp':1,'fsdp':1}],
                    [1,4,2,{'tp':0,'fsdp':0}],[1,4,2,{'tp':1,'fsdp':0}],[1,4,2,{'tp':0,'fsdp':1}],[1,4,2,{'tp':1,'fsdp':1}],
                    [1,8,1,{}],
                    [2,1,4,{'fsdp':0}],[2,1,4,{'fsdp':1}],
                    [2,2,2,{'tp':0,'fsdp':0}],[2,2,2,{'tp':1,'fsdp':0}],[2,2,2,{'tp':0,'fsdp':1}],[2,2,2,{'tp':1,'fsdp':1}],
                    [2,4,1,{}],
                    [4,1,2,{'fsdp':0}],[4,1,2,{'fsdp':1}],
                    [4,2,1,{}],
                    [8,1,1,{}]]
    elif args.type == 'dp+tp':
        # only dp+tp
        strategies = [[1,1,8,{'fsdp':0}],
                    [1,2,4,{'tp':0,'fsdp':0}],[1,2,4,{'tp':1,'fsdp':0}],
                    [1,4,2,{'tp':0,'fsdp':0}],[1,4,2,{'tp':1,'fsdp':0}],
                    [1,8,1,{}]]
    elif args.type == 'dp+pp':
        # only dp+pp
        strategies = [[1,1,8,{'fsdp':0}],
                    [2,1,4,{'fsdp':0}],
                    [4,1,2,{'fsdp':0}],
                    [8,1,1,{}]]

    # Load profiling configs
    hidden_size = args.hidden_size
    layer_num = args.layer_num
    gpu_num = args.gpu_num
    comm_coe_dict, overlap_coe, fwd_time = read_profiling_configs(gpu_num)
    fwd_time = fwd_time['fwd_time_hidden_%d'%hidden_size]

    print('================================================================================')
    print('------- Model configs -------')
    print('Layer_num:', layer_num)
    print('Hidden_size:', hidden_size)
    print('================================================================================')
    print('--- Optimization configs ----')
    print('Memory_constraint: %d GB'%args.memory_constraint)
    print('Optimization type:', args.type)
    print('================================================================================')
    print('---- Environment configs ----')
    print('Allreduce comm_coe dict (ms/MB):', comm_coe_dict)
    print('Overlap coefficient:', overlap_coe)
    print('--- Model forward configs ---')
    print('Forward computation time (ms/layer/bsz):', fwd_time)
    print('================================================================================')

    def optimal_chunk_func(local_bsz, strategy):
        if strategy[0] == 1:
            return 1
        if np.array_equal(strategy, [2,4,1]) or np.array_equal(strategy, [4,2,1]):
            return local_bsz/4
        if local_bsz < 6:
            return local_bsz/2
        else:
            return local_bsz/8+2

    if hidden_size == 1024:
        # Bert Large 1024 config
        parameter_size = 48.05
        forward_compute_time_per_layer = fwd_time
        tp_activation_per_bsz_dict = {  1:85.00833, 
                                        2:47.00833, 
                                        4:28.008125, 
                                        8:18.54875}
        memcost_model_args = {  'parameter_size': parameter_size,
                                'tp_activation_per_bsz_dict': tp_activation_per_bsz_dict,
                                'other_model_states': 640,
                                'other_activation_per_bsz': 300}
        timecost_model_args_with_overlap = { 
                                'parameter_size': parameter_size,
                                'microbatch': True,
                                'optimal_chunk_func': optimal_chunk_func,
                                'sequence_length': 512,
                                'hidden_size': 1024,
                                'forward_computation_time': forward_compute_time_per_layer,
                                'bct_fct_coe': 2,
                                'extra_overhead': 0,
                                'comm_coe_dict': comm_coe_dict,
                                'dp_overlap_coe': overlap_coe,
                                'bct_overlap_coe': overlap_coe}
    elif hidden_size == 1280:
        # Bert Large 1280 config
        parameter_size = 76.98
        forward_compute_time_per_layer = fwd_time
        tp_activation_per_bsz_dict = {  1:94.07, 
                                        2:52.475, 
                                        4:31.754375, 
                                        8:21.61171875}
        memcost_model_args = {  'parameter_size': parameter_size,
                                'tp_activation_per_bsz_dict': tp_activation_per_bsz_dict,
                                'other_model_states': 790,
                                'other_activation_per_bsz': 300}
        timecost_model_args_with_overlap = { 
                                'parameter_size': parameter_size,
                                'microbatch': True,
                                'optimal_chunk_func': optimal_chunk_func,
                                'sequence_length': 512,
                                'hidden_size': 1280,
                                'forward_computation_time': forward_compute_time_per_layer,
                                'bct_fct_coe': 2,
                                'extra_overhead': 0,
                                'comm_coe_dict': comm_coe_dict,
                                'dp_overlap_coe': overlap_coe,
                                'bct_overlap_coe': overlap_coe}

    search_history = dict()
    def search(max_mem):
        bsz_scale = 8
        bsz_start = bsz_scale if args.search_from_min_bsz else estimate_bsz_start(bsz_scale)
        print('Searching batch_size start from: %d, batch_size scale: %d'%(bsz_start, bsz_scale))
        print("----Searching with max memory %d MB----"%max_mem)
        dp_on_model = DpOnModel(strategies, 
                                MemoryCostModel, 
                                TimeCostModel_with_overlap, 
                                memcost_model_args,
                                timecost_model_args_with_overlap,
                                max_mem=max_mem,
                                search_history=search_history,
                                layer_num=layer_num,
                                comm_coe_dict=comm_coe_dict)

        results = dict()
        max_throughput, optimal_bsz, max_bsz = -1, -1, -1
        for bsz in range(bsz_start, 1024, bsz_scale):
            print("****Testing with bsz=", bsz, "****")
            min_cost, min_res_list, min_pp_deg, mem_remain, mem_cost = dp_on_model.fit(bsz)
            throughput = bsz / min_cost
            print(f"[Optimal pp_deg={min_pp_deg}] Minimized timecost={min_cost} Memory remaining={mem_remain} Memory cost={mem_cost}")
            print(f"Max throughput={throughput} samples/s")
            print_strategies(min_res_list)
            results[bsz] = {'min_cost': min_cost, 'min_res_list': min_res_list, 'min_pp_deg': min_pp_deg, 
                            'mem_remain': mem_remain, 'mem_cost': mem_cost, 'throughput': throughput}
            if throughput > max_throughput:
                max_throughput = throughput
                optimal_bsz = bsz
            if min_pp_deg == -1:
                break
            max_bsz = bsz

        print('\nFinal results of max memory %d MB:'%max_mem)
        re = results[optimal_bsz]
        print(f"Optimal bsz = {optimal_bsz} Max throughput={re['throughput']} samples/s")
        print(f"pp_deg={re['min_pp_deg']} Minimized timecost={re['min_cost']} Memory remaining={re['mem_remain']} Memory cost={re['mem_cost']}")
        print_strategies(re['min_res_list'])

        if re['min_pp_deg'] > 0 and re['min_res_list'] is not None:
            config = strategy2config(re['min_res_list']*re['min_pp_deg'])
            config['global_bsz'] = optimal_bsz
            config['chunks'] = max([int(optimal_chunk_func(optimal_bsz//s[2],s)) for s in re['min_res_list']]) if config['pp_deg'] > 1 else 1
            file_name = './configs/galvatron_config_%dgpus_%dhidden_%dlayers_%dG_%s.json'%(gpu_num,hidden_size,layer_num,max_mem//1024,args.type)
            write_json_config(config, file_name)
            print('Already written optimized parallelism config into galvatron config file %s!'%(file_name))

        if max_bsz > -1 and max_bsz != optimal_bsz:
            re = results[max_bsz]
            print(f"\nMax bsz = {max_bsz} Max throughput={re['throughput']} samples/s")
            print(f"pp_deg={re['min_pp_deg']} Minimized timecost={re['min_cost']} Memory remaining={re['mem_remain']} Memory cost={re['mem_cost']}")
            print_strategies(re['min_res_list'])
        print("-----------------------------------------")

    # Check cost model
    def check_cost_model():
        bsz = 8
        for strategy in strategies:
            re = MemoryCostModel(strategy, global_batch_size=bsz, **memcost_model_args).get_memory_cost()
            print(form_strategy(strategy), re['enc_total'], re['other'], re['enc_total']*24/strategy[0]+re['other']-1024)
        print()
        for strategy in strategies:
            re = TimeCostModel_with_overlap(strategy, global_batch_size=bsz, **timecost_model_args_with_overlap).gen_result()
            print(form_strategy(strategy), re*24)
        print()

    def estimate_bsz_start(scale):
        def estimate_strategy_max_bsz(s):
            dp_on_model = DpOnModel(s, MemoryCostModel, TimeCostModel_with_overlap, 
                                    memcost_model_args, timecost_model_args_with_overlap,
                                    max_mem=max_mem, layer_num=layer_num, comm_coe_dict=comm_coe_dict)
            max_bsz = 0
            for bsz in range(scale, 1024, scale):
                min_cost, min_res_list, min_pp_deg, mem_remain, mem_cost = dp_on_model.fit(bsz, False)
                if min_pp_deg == -1:
                    max_bsz = bsz - scale
                    break
            return max_bsz
        bsz_start = estimate_bsz_start_8gpus(args.type,scale,estimate_strategy_max_bsz)
        return bsz_start

    # check_cost_model()
    mem_list = [8, 12, 16, 20]
    if args.memory_constraint > 0:
        mem_list = [args.memory_constraint]
    mem_list = [mem * 1024 for mem in mem_list]
    for max_mem in mem_list:
        search(max_mem)
        print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hidden_size", type=int, default=1280, help="Hidden size of transformer model", choices=[1024, 1280],
    )
    parser.add_argument(
        "--layer_num", type=int, default=24, help="Number of layers"
    )
    parser.add_argument(
        "--gpu_num", type=int, default=8, help="Number of GPUs",
    )
    parser.add_argument(
        "--memory_constraint", type=int, default=8, help="Memory constraint of Galvatron",
    )
    parser.add_argument(
        "--type", type=str, default='full', help="Galvatron parallelism optimization type.", choices=['full','dp+tp','dp+pp'],
    )
    parser.add_argument(
        "--search_from_min_bsz", type=int, default=0, help="If 0, start searching from a recommended bsz to accelerate optimization.",
    )
    args = parser.parse_args()
    parallelism_optimization(args)