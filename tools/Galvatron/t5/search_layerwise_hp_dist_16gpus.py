import sys
sys.path.insert(0, '..')
from utils import MemoryCostModelDist, TimeCostModelDist_with_overlap
from utils import DpOnModel_dist, print_strategies, form_strategy
import numpy as np
from utils import strategy2config, read_json_config, write_json_config, read_allreduce_bandwidth_config, read_p2p_bandwidth_config, array2str, estimate_bsz_start_16gpus
import argparse

def read_profiling_configs(gpu_num):
    allreduce_bandwidth_config_path = '../env_configs/allreduce_bandwidth_dist_%d_gpus.json'%gpu_num
    comm_coe_dict = read_allreduce_bandwidth_config(allreduce_bandwidth_config_path, gpu_num=gpu_num)
    overlap_coe_path = '../env_configs/overlap_coefficient.json'
    overlap_coe = read_json_config(overlap_coe_path)['overlap_coe']
    fwd_profiling_path = './configs/forward_profiling_config.json'
    fwd_time = read_json_config(fwd_profiling_path)
    env_config_path = '../env_configs/p2p_bandwidth_dist_%d_gpus.json'%gpu_num
    p2p_comm_coe_dict = read_p2p_bandwidth_config(env_config_path, gpu_num=gpu_num)
    return comm_coe_dict, overlap_coe, fwd_time, p2p_comm_coe_dict

def parallelism_optimization(args):
    if args.type == 'full':
        # full strategies
        strategies = [[1,1,16,{'fsdp':0}],[1,1,16,{'fsdp':1}],
                    [1,2,8,{'tp':0,'fsdp':0}],[1,2,8,{'tp':1,'fsdp':0}],[1,2,8,{'tp':0,'fsdp':1}],[1,2,8,{'tp':1,'fsdp':1}],
                    [1,4,4,{'tp':0,'fsdp':0}],[1,4,4,{'tp':1,'fsdp':0}],[1,4,4,{'tp':0,'fsdp':1}],[1,4,4,{'tp':1,'fsdp':1}],
                    [1,8,2,{'tp':0,'fsdp':0}],[1,8,2,{'tp':1,'fsdp':0}],[1,8,2,{'tp':0,'fsdp':1}],[1,8,2,{'tp':1,'fsdp':1}],
                    [1,16,1,{}],
                    [2,1,8,{'fsdp':0}],[2,1,8,{'fsdp':1}],
                    [2,2,4,{'tp':0,'fsdp':0}],[2,2,4,{'tp':1,'fsdp':0}],[2,2,4,{'tp':0,'fsdp':1}],[2,2,4,{'tp':1,'fsdp':1}],
                    [2,4,2,{'tp':0,'fsdp':0}],[2,4,2,{'tp':1,'fsdp':0}],[2,4,2,{'tp':0,'fsdp':1}],[2,4,2,{'tp':1,'fsdp':1}],
                    [2,8,1,{}],
                    [4,1,4,{'fsdp':0}],[4,1,4,{'fsdp':1}],
                    [4,2,2,{'tp':0,'fsdp':0}],[4,2,2,{'tp':1,'fsdp':0}],[4,2,2,{'tp':0,'fsdp':1}],[4,2,2,{'tp':1,'fsdp':1}],
                    [4,4,1,{}],
                    [8,1,2,{'fsdp':0}],[8,1,2,{'fsdp':1}],
                    [8,2,1,{}],
                    [16,1,1,{}]]
    elif args.type == 'dp+tp':
        # only dp+tp
        strategies = [[1,1,16,{'fsdp':0}],
                    [1,2,8,{'tp':0,'fsdp':0}],[1,2,8,{'tp':1,'fsdp':0}],
                    [1,4,4,{'tp':0,'fsdp':0}],[1,4,4,{'tp':1,'fsdp':0}],
                    [1,8,2,{'tp':0,'fsdp':0}],[1,8,2,{'tp':1,'fsdp':0}],
                    [1,16,1,{}]]
    elif args.type == 'dp+pp':
        # only dp+pp
        strategies = [[1,1,16,{'fsdp':0}],
                    [2,1,8,{'fsdp':0}],
                    [4,1,4,{'fsdp':0}],
                    [8,1,2,{'fsdp':0}],
                    [16,1,1,{}]]

    # Load profiling configs
    hidden_size = args.hidden_size
    layer_num_list = [args.layer_num_encoder, args.layer_num_decoder]
    gpu_num = args.gpu_num
    comm_coe_dict, overlap_coe, fwd_time, p2p_comm_coe_dict = read_profiling_configs(gpu_num)
    fwd_time_enc = fwd_time['fwd_time_hidden_%d'%hidden_size]['encoder']
    fwd_time_dec = fwd_time['fwd_time_hidden_%d'%hidden_size]['decoder']

    print('================================================================================')
    print('------- Model configs -------')
    print('Layer_nums:', layer_num_list)
    print('Hidden_size:', hidden_size)
    print('================================================================================')
    print('--- Optimization configs ----')
    print('Memory_constraint: %d GB'%args.memory_constraint)
    print('Optimization type:', args.type)
    print('================================================================================')
    print('---- Environment configs ----')
    print('Allreduce comm_coe dict (ms/MB):', comm_coe_dict)
    print('P2P comm_coe dict (ms/MB):', p2p_comm_coe_dict)
    print('Overlap coefficient:', overlap_coe)
    print('--- Model forward configs ---')
    print('Forward computation time:', fwd_time)
    print('================================================================================')

    def optimal_chunk_func(local_bsz, strategy):
        local_bsz = local_bsz // strategy[1]
        re = np.ceil(local_bsz / 4)
        re = 1 if re == 0 else re
        return re

    microbatch = True

    other_memory_pp_off = {'model_states': 940, 'activation': 700}
    other_memory_pp_on = {'first_stage':{'model_states': 504, 'activation': 250}, 'last_stage':{'model_states': 630, 'activation': 270}}

    if hidden_size == 1024:
        # T5 Large 1024 Encoder config
        parameter_size_enc = 48.01
        forward_compute_time_per_layer = fwd_time_enc
        tp_activation_per_bsz_dict_enc = {  1:91.003, 
                                        2:54.004, 
                                        4:32.785, 
                                        8:22.8175,
                                        16:17.8}
        memcost_model_args_enc = {  'parameter_size': parameter_size_enc,
                                'tp_activation_per_bsz_dict': tp_activation_per_bsz_dict_enc,
                                'other_memory_pp_off': other_memory_pp_off,
                                'other_memory_pp_on': other_memory_pp_on,
                                'peak_reduction_with_chunks': 140,
                                'microbatch': microbatch,
                                'optimal_chunk_func': optimal_chunk_func,
                                'model_type': 't5'}
        timecost_model_args_with_overlap_enc = { 
                                'parameter_size': parameter_size_enc,
                                'microbatch': microbatch,
                                'optimal_chunk_func': optimal_chunk_func,
                                'sequence_length': 512,
                                'hidden_size': 1024,
                                'forward_computation_time': forward_compute_time_per_layer,
                                'bct_fct_coe': 2,
                                'extra_overhead': 60,
                                'comm_coe_dict': comm_coe_dict,
                                'dp_overlap_coe': overlap_coe,
                                'bct_overlap_coe': overlap_coe,
                                'p2p_comm_coe_dict': p2p_comm_coe_dict,
                                'layer_type': 'enc'}

        # T5 Large 1024 Decoder config
        parameter_size_dec = 64.012
        forward_compute_time_per_layer = fwd_time_dec
        tp_activation_per_bsz_dict_dec = {  1:157.755, 
                                        2:90.756, 
                                        4:56.384, 
                                        8:39.117,
                                        16:31.1}
        memcost_model_args_dec = {  'parameter_size': parameter_size_dec,
                                'tp_activation_per_bsz_dict': tp_activation_per_bsz_dict_dec,
                                'other_memory_pp_off': other_memory_pp_off,
                                'other_memory_pp_on': other_memory_pp_on,
                                'peak_reduction_with_chunks': 140,
                                'microbatch': microbatch,
                                'optimal_chunk_func': optimal_chunk_func,
                                'model_type': 't5'}
        timecost_model_args_with_overlap_dec = { 
                                'parameter_size': parameter_size_dec,
                                'microbatch': microbatch,
                                'optimal_chunk_func': optimal_chunk_func,
                                'sequence_length': 512,
                                'hidden_size': 1024,
                                'forward_computation_time': forward_compute_time_per_layer,
                                'bct_fct_coe': 2,
                                'extra_overhead': 60,
                                'comm_coe_dict': comm_coe_dict,
                                'dp_overlap_coe': overlap_coe,
                                'bct_overlap_coe': overlap_coe,
                                'p2p_comm_coe_dict': p2p_comm_coe_dict,
                                'layer_type': 'dec'}

    memcost_model_args=[memcost_model_args_enc, memcost_model_args_dec]
    timecost_model_args=[timecost_model_args_with_overlap_enc, timecost_model_args_with_overlap_dec]

    def pp_stage_divide_greedy(memcost_model_args, layer_num, pp_deg, bsz, strategies):
        assert(len(memcost_model_args)==len(layer_num))
        if pp_deg == 1:
            return [np.sum(layer_num)], None
        layer_type_num = len(layer_num)
        layer_min_memcost = []
        strategies = list(filter(lambda s: s[0] == pp_deg, strategies))
        if len(strategies)==0:
            return None, None
        for i in range(layer_type_num):
            memcosts = [MemoryCostModelDist(strategy, global_batch_size=bsz, **memcost_model_args[i]).get_memory_cost()['enc_total'] for strategy in strategies]
            layer_min_memcost.append(np.min(memcosts))
        other_cost = MemoryCostModelDist(strategies[0], global_batch_size=bsz, **memcost_model_args[0]).get_memory_cost()['other']
        #print(layer_min_memcost, other_cost)
        min_memcost_all_layers = []
        for i in range(layer_type_num):
            min_memcost_all_layers += [layer_min_memcost[i]]*layer_num[i]
        #print(min_memcost_all_layers)
        avg_mem_cost = (np.sum(min_memcost_all_layers)+np.sum(other_cost))/pp_deg
        #print('Avg memcost:', avg_mem_cost)

        pp_divide = [0]*pp_deg
        mem_cost_per_stage = other_cost.copy()
        idx = len(min_memcost_all_layers)-1
        for i in range(pp_deg-1,-1,-1):
            while True:
                if idx < 0:
                    break
                if i > 0 and avg_mem_cost - mem_cost_per_stage[i] < 0.5 * min_memcost_all_layers[idx]:
                    break
                else:
                    mem_cost_per_stage[i]+=min_memcost_all_layers[idx]
                    idx-=1
                    pp_divide[i]+=1
        # print(pp_divide)

        # Avoid too much memory cost on previous stages
        for i in range(pp_deg-1):
            left, right = int(np.sum(pp_divide[:i])), int(np.sum(pp_divide[:i+1]))
            mem_cost_cur_stage = np.sum(min_memcost_all_layers[left:right]) + other_cost[i]
            while mem_cost_cur_stage > avg_mem_cost * 1.3:
                pp_divide[i] -= 1
                pp_divide[i+1] += 1
                right -= 1
                mem_cost_cur_stage -= min_memcost_all_layers[right]

        # Avoid no layers on previous stages
        for i in range(pp_deg-1):
            while pp_divide[i] <= 0:
                pp_divide[i] += 1
                pp_divide[i+1] -= 1

        # Avoid no layers on last stage
        for i in range(pp_deg-1, 0, -1):
            while pp_divide[i] <= 0:
                pp_divide[i] += 1
                pp_divide[i-1] -= 1
        
        mem_cost_per_stage_adjusted = other_cost.copy()
        # print(pp_divide)
        # print(other_cost, avg_mem_cost)
        for i in range(pp_deg):
            left, right = int(np.sum(pp_divide[:i])), int(np.sum(pp_divide[:i+1]))
            mem_cost_per_stage_adjusted[i] +=  np.sum(min_memcost_all_layers[left:right])
        # print(mem_cost_per_stage,mem_cost_per_stage_adjusted)
        return pp_divide, mem_cost_per_stage_adjusted

    def get_pp_stages_for_all_bsz():
        bszs = list(range(8, 128, 8))
        pp_stage_dict_for_bsz = dict()
        for bsz in bszs:
            pp_stage_dict = dict()
            for pp_deg in [1,2,4,8,16]:
                pp_divide, mem_cost_per_stage = pp_stage_divide_greedy(memcost_model_args, layer_num_list, pp_deg, bsz, strategies)
                #print(bsz, pp_deg, pp_divide, mem_cost_per_stage)
                pp_stage_dict[pp_deg] = pp_divide
            pp_stage_dict_for_bsz[bsz] = pp_stage_dict
        return pp_stage_dict_for_bsz


    search_history = dict()
    def search(max_mem):
        bsz_scale = 8
        bsz_scale = 16 if args.type == 'dp+tp' and bsz_scale < 16 else bsz_scale
        bsz_start = bsz_scale if args.search_from_min_bsz else estimate_bsz_start(bsz_scale)
        print('Searching batch_size start from: %d, batch_size scale: %d'%(bsz_start, bsz_scale))
        print("----Searching with max memory %d MB----"%max_mem)
        results = dict()
        max_throughput, optimal_bsz, max_bsz = -1, -1, -1
        for bsz in range(bsz_start, 1024, bsz_scale):
            pp_stage_dict = pp_stage_dict_for_bsz[bsz]
            dp_on_model = DpOnModel_dist(strategies, 
                                    MemoryCostModelDist, 
                                    TimeCostModelDist_with_overlap, 
                                    memcost_model_args=memcost_model_args,
                                    timecost_model_args=timecost_model_args,
                                    max_mem=max_mem,
                                    layer_num =layer_num_list,
                                    multi_layer_type = True,
                                    pp_stage_dict = pp_stage_dict,
                                    search_history=search_history,
                                    comm_coe_dict=comm_coe_dict,
                                    gpu_num=gpu_num)
            
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
            result_strategy = []
            if isinstance(re['min_res_list'],list):
                for l in re['min_res_list']:
                    result_strategy += l
            else:
                result_strategy = re['min_res_list']
            config = strategy2config(result_strategy)
            config['global_bsz'] = optimal_bsz
            config['chunks'] = max([int(optimal_chunk_func(optimal_bsz//s[2],s)) for s in result_strategy]) if config['pp_deg'] > 1 else 1
            config['pp_division'] = array2str(pp_stage_dict_for_bsz[optimal_bsz][config['pp_deg']])
            file_name = './configs/galvatron_config_%dgpus_%dhidden_%denc_%ddec_%dG_%s.json'%(gpu_num,hidden_size,args.layer_num_encoder,args.layer_num_decoder,max_mem//1024,args.type)
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
        bsz = 16
        layer_num = 24
        mem_0, mem_1, other = [], [], []
        for strategy in strategies:
            re = MemoryCostModelDist(strategy, global_batch_size=bsz, **memcost_model_args_enc).get_memory_cost()
            print(form_strategy(strategy), re['enc_total'], re['other'])
            mem_0.append(re['enc_total'])
            other.append(re['other'])
        print()
        for strategy in strategies:
            re = MemoryCostModelDist(strategy, global_batch_size=bsz, **memcost_model_args_dec).get_memory_cost()
            print(form_strategy(strategy), re['enc_total'], re['other'])
            mem_1.append(re['enc_total'])
        print()
        for i in range(len(strategies)):
            strategy = strategies[i]
            if strategy[0]==1:
                print(form_strategy(strategy), mem_0[i]*layer_num+mem_1[i]*layer_num+other[i][0]-1024)
            else:
                print(form_strategy(strategy), mem_0[i]*layer_num//strategy[0]*2+other[i][0]-1024, mem_1[i]*layer_num//strategy[0]*2+other[i][-1]-1024)
        print()

        enc_re = []
        for strategy in strategies:
            re = TimeCostModelDist_with_overlap(strategy, global_batch_size=bsz, **timecost_model_args_with_overlap_enc).gen_result()
            print(form_strategy(strategy), re*layer_num)
            enc_re.append(re*layer_num)
        print()
        dec_re=[]
        for strategy in strategies:
            re = TimeCostModelDist_with_overlap(strategy, global_batch_size=bsz, **timecost_model_args_with_overlap_dec).gen_result()
            print(form_strategy(strategy), re*layer_num)
            dec_re.append(re*layer_num)
        print()
        for i in range(len(strategies)):
            print(form_strategy(strategies[i]), enc_re[i]+dec_re[i])
        print()

    def estimate_bsz_start(scale):
        def estimate_strategy_max_bsz(s):
            max_bsz = 0
            scale_ = 16 if s[0][0] == 1 and scale < 16 else scale
            for bsz in range(scale_, 1024, scale_):
                pp_stage_dict = pp_stage_dict_for_bsz[bsz]
                dp_on_model = DpOnModel_dist(s, MemoryCostModelDist, TimeCostModelDist_with_overlap, 
                                        memcost_model_args, timecost_model_args,
                                        max_mem=max_mem, layer_num=layer_num_list, 
                                        multi_layer_type = True, pp_stage_dict = pp_stage_dict,
                                        comm_coe_dict=comm_coe_dict, gpu_num=gpu_num)
                min_cost, min_res_list, min_pp_deg, mem_remain, mem_cost = dp_on_model.fit(bsz, False)
                if min_pp_deg == -1:
                    max_bsz = bsz - scale_
                    break
            return max_bsz
        bsz_start = estimate_bsz_start_16gpus(args.type,scale,estimate_strategy_max_bsz)
        return bsz_start

    # check_cost_model()
    pp_stage_dict_for_bsz = get_pp_stages_for_all_bsz()
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
        "--hidden_size", type=int, default=1024, help="Hidden size of transformer model", choices=[1024],
    )
    parser.add_argument(
        "--layer_num_encoder", type=int, default=24, help="Number of layers"
    )
    parser.add_argument(
        "--layer_num_decoder", type=int, default=24, help="Number of layers"
    )
    parser.add_argument(
        "--gpu_num", type=int, default=16, help="Number of GPUs",
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