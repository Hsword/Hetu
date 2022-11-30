import numpy as np
from tqdm import trange

def form_strategy(strategy):
    template = '%d-%s-%s'
    assert len(strategy) == 4
    info = strategy[-1]
    pp_deg = strategy[0]
    tp_deg = '%d'%strategy[1]
    dp_deg = '%d'%strategy[2]
    if 'fsdp' in info.keys():
        if info['fsdp']:
            dp_deg += 'f'
    if 'tp' in info.keys():
        if info['tp']:
            tp_deg += '*'
        else:
            dp_deg += '*'
    return template%(pp_deg, tp_deg, dp_deg)

def print_strategies(strategy_list):
    if strategy_list is None:
        print(strategy_list)
        return
    if isinstance(strategy_list[0][0],list):
        result_list = []
        for sub_strategy_list in strategy_list:
            sub_result_list = []
            for strategy in sub_strategy_list:
                sub_result_list.append(form_strategy(strategy))
            result_list.append(', '.join(sub_result_list))
        print(' || '.join(result_list))
    else:
        result_list = []
        for strategy in strategy_list:
            result_list.append(form_strategy(strategy))
        print(', '.join(result_list))

def estimate_bsz_start_16gpus(type,scale,max_bsz_estimator):
    prune_percent = 0.65
    if type == 'full':
        baselines = [[1,1,16,{'fsdp':0}],[1,16,1,{'fsdp':0}],[16,1,1,{}],[1,1,16,{'fsdp':1}]]
    elif type == 'dp+tp':
        baselines = [[1,1,16,{'fsdp':0}],[1,16,1,{'fsdp':0}]]
    elif type == 'dp+pp':
        baselines = [[1,1,16,{'fsdp':0}],[16,1,1,{}]]
        prune_percent = 0
    max_bsz_baselines = [max_bsz_estimator([s]) for s in baselines]
    # print(max_bsz_baselines)
    max_bsz, min_bsz = np.max(max_bsz_baselines), np.min(max_bsz_baselines)
    bsz_start = int((min_bsz*(1-prune_percent)+max_bsz*prune_percent)//scale*scale)
    bsz_start = bsz_start if bsz_start > scale else scale
    return bsz_start

class DPAlg():
    def __init__(self, max_mem=8200, layer_num=24, strategy_num=4) -> None:
        self.max_mem = max_mem + 1
        self.layer_num = layer_num
        self.strategy_num = strategy_num

        self._f = np.full((self.max_mem, strategy_num), 0, dtype=np.float64)
        
        self.v_data = None
        self.inter_cost = None
        self.intra_cost = None

        self._mark = np.full((layer_num, self.max_mem, strategy_num), -1)

    
    def set_v_and_cost(self, v: np.ndarray, intra_layer_cost: np.ndarray, inter_layer_cost: np.ndarray):
        assert v.ndim == 2
        assert inter_layer_cost.ndim == 3
        assert intra_layer_cost.ndim == 2

        assert v.shape[0] == self.layer_num
        assert v.shape[1] == self.strategy_num

        assert inter_layer_cost.shape[0] == self.layer_num
        assert inter_layer_cost.shape[1] == self.strategy_num and inter_layer_cost.shape[2] == self.strategy_num

        assert intra_layer_cost.shape[0] == self.layer_num
        assert intra_layer_cost.shape[1] == self.strategy_num

        self.v_data = v
        self.inter_cost = inter_layer_cost
        self.intra_cost = intra_layer_cost

    def fit(self):
        if self.strategy_num == 1:
            total_v = np.sum(self.v_data[:,0])
            total_cost = np.sum(self.intra_cost[:,0])
            if total_v <= self.max_mem - 1:
                return total_cost, [0] * self.layer_num, self.max_mem - 1 - total_v
            else:
                return np.inf, None, -1

        for i in range(self.layer_num):
            for v in range(self.max_mem - 1, -1, -1):
                for s in range(self.strategy_num):

                    if v < self.v_data[i, s]:
                        self._mark[i, v, s] = -1
                        self._f[v, s] = np.inf
                        continue

                    candidates = [self._f[v - self.v_data[i, s], si] + self.inter_cost[i, si, s] for si in range(self.strategy_num)]
                    candidates = np.array(candidates) + self.intra_cost[i, s]

                    min_index = np.argmin(candidates)

                    self._mark[i, v, s] = min_index
                    self._f[v, s] = candidates[min_index]
        
        next_index, next_v = np.argmin(self._f[-1, :]), self.max_mem - 1
        total_cost = self._f[-1, next_index]

        if not total_cost < np.inf:
            return np.inf, None, -1

        res_list = [-1] * self.layer_num
        res_list[-1] = next_index

        for i in range(self.layer_num - 1, 0, -1):
            next_index, next_v = self._mark[i, next_v, next_index], next_v - self.v_data[i, next_index]
            res_list[i - 1] = next_index

        return total_cost, res_list, next_v - self.v_data[0, next_index]

class DpOnModel_dist:
    def __init__(   self, 
                    strategies_set, 
                    memcost_model, 
                    timecost_model, 
                    memcost_model_args,
                    timecost_model_args,
                    max_mem=8192, 
                    layer_num=24,
                    multi_layer_type=False,
                    pp_stage_dict=None,
                    search_history=None,
                    comm_coe_dict={},
                    gpu_num=8):
        self.strategies_set = strategies_set
        self.memcost_model = memcost_model
        self.timecost_model = timecost_model
        self.memcost_model_args = memcost_model_args
        self.timecost_model_args = timecost_model_args
        self.max_mem = max_mem
        self.layer_num = layer_num
        self.n_gpu = strategies_set[0][0] * strategies_set[0][1] * strategies_set[0][2]
        self.ppdeg_set = np.unique(np.array([s[0] for s in strategies_set], dtype=np.int32))
        self.multi_layer_type = multi_layer_type
        self.search_history = search_history
        self.comm_coe_dict = comm_coe_dict
        self.gpu_num = gpu_num
        if multi_layer_type:
            # If multi_layer_type == True, layer_num/memcost_model_args/timecost_model_args should be list.
            # e.g. for T5, layer_num = [12, 12], memcost_model_args = [memcost_model_args_enc, memcost_model_args_dec]
            # timecost_model_args = [timecost_model_args_enc, timecost_model_args_dec]
            # pp_stage_dict = {1:[24], 2: [15, 9], 4: [7, 7, 5, 5], 8:[4, 4, 4, 4, 2, 2, 2, 2]}
            assert(isinstance(layer_num, list))
            self.total_layer_num = sum(layer_num)
            assert(isinstance(memcost_model_args, list) and len(layer_num) == len(memcost_model_args))
            assert(isinstance(timecost_model_args, list) and len(layer_num) == len(timecost_model_args))
            assert(isinstance(pp_stage_dict, dict))
            for ppdeg in self.ppdeg_set:
                if ppdeg > 1:
                    assert(ppdeg in pp_stage_dict.keys())
                    assert(sum(pp_stage_dict[ppdeg])==self.total_layer_num)
            self.pp_stage_dict = pp_stage_dict
            if 1 not in self.pp_stage_dict.keys():
                self.pp_stage_dict[1] = [self.total_layer_num]


    def _build_dp_and_run(self, pp_deg, bsz):
        # Look for results in search history
        key = (bsz, pp_deg)
        from_history = False
        if self.search_history is not None and key in self.search_history.keys() and self.search_history[key]['mem_cost'] <= self.max_mem:
            re = self.search_history[key]
            comm_cost, res_list, mem_remain, mem_cost = \
                re['comm_cost'], re['res_list'], self.max_mem-re['mem_cost'], re['mem_cost']
            best_strategy_flag, from_history = True, True
            return comm_cost, res_list, mem_remain, mem_cost, best_strategy_flag, from_history

        strategy_set = list(filter(lambda s: s[0] == pp_deg, self.strategies_set))
        strategy_num = len(strategy_set)
        layer_num = self.layer_num // pp_deg

        intra_layer_cost = [self.timecost_model(strategy, bsz, **self.timecost_model_args).gen_result() for strategy in strategy_set]
        intra_layer_cost = np.array(intra_layer_cost, dtype=np.float64).reshape(1, -1).repeat(layer_num, axis=0)
        min_cost_strategy_ids = np.argmin(intra_layer_cost, axis=1)

        mem_cost_list = [self.memcost_model(strategy, bsz, **self.memcost_model_args).get_memory_cost() for strategy in strategy_set]
        other_mem_cost = int(np.ceil(np.max(mem_cost_list[0]['other'])))
        v = [cost['enc_total'] for cost in mem_cost_list]
        v = np.ceil(np.array(v)).astype(np.int32)
        v = v.reshape(1, -1).repeat(layer_num, axis=0)

        inter_layer_cost = np.zeros((strategy_num, strategy_num))
        for i in range(strategy_num):
            for j in range(strategy_num):
                case1 = strategy_set[j][1] > strategy_set[i][1]
                case2 = False
                case3 = False
                if 'tp' in strategy_set[j][-1].keys() and 'tp' in strategy_set[i][-1].keys():
                    case2 = (strategy_set[j][1] == strategy_set[i][1] and strategy_set[j][-1]['tp'] != strategy_set[i][-1]['tp'])
                    world_size = strategy_set[i][1] * strategy_set[i][2]
                    case3 = ( strategy_set[j][1] < strategy_set[i][1] and strategy_set[j][-1]['tp'] != strategy_set[i][-1]['tp'] and strategy_set[j][1] > 1 and strategy_set[i][1] < world_size)
                if case1 or case2 or case3:
                     ratio = strategy_set[j][1]
                     activation = 2 * bsz / strategy_set[j][2]
                     inter_layer_cost[i, j] = (ratio - 1) * activation / ratio

        # find corresponding communication coefficient
        for i in range(strategy_num):
            for j in range(strategy_num):
                tp_size, dp_size = strategy_set[j][1], strategy_set[j][2]
                if tp_size == 1 or dp_size == 1:
                    coe = self.comm_coe_dict[pp_deg]['%d'%tp_size]
                else:
                    # In this case, strategy[-1]['tp'] represents tp_consecutive_flag
                    info = strategy_set[j][-1]
                    assert 'tp' in info.keys() and info['tp'] in [0, 1]
                    if info['tp']:
                        coe = self.comm_coe_dict[pp_deg]['%d_1'%tp_size]
                    else:
                        coe = self.comm_coe_dict[pp_deg]['%d_0'%tp_size]
                inter_layer_cost[i, j] = inter_layer_cost[i, j] * coe * 0.001

                # add a small bias to sort fsdp and dp
                strategy0, strategy1 = strategy_set[i], strategy_set[j]
                if i != j and np.array_equal(strategy0[:3], strategy1[:3]):
                    case1 = 'tp' not in strategy0[-1] and strategy0[-1]['fsdp']!=strategy1[-1]['fsdp']
                    case2 = 'tp' in strategy0[-1] and strategy0[-1]['tp']==strategy1[-1]['tp'] and strategy0[-1]['fsdp']!=strategy1[-1]['fsdp']
                    if (case1 or case2) and strategy0[-1]['fsdp']:
                        inter_layer_cost[i, j] = 1e-4

        inter_layer_cost = np.expand_dims(inter_layer_cost, axis=0).repeat(layer_num, axis=0)
        inter_layer_cost[0, :, :] = 0 # no inter-layer communication cost in first layer

        if self.max_mem - other_mem_cost <= 0:
            return np.inf, None, -1, np.inf, False, False
        
        dp = DPAlg(self.max_mem - other_mem_cost, layer_num, strategy_num)
        dp.set_v_and_cost(v, intra_layer_cost, inter_layer_cost)

        comm_cost, res_list, mem_remain = dp.fit()
        best_strategy_flag = res_list is not None and (np.array(res_list) == min_cost_strategy_ids).all()
        if res_list is not None:
            res_list = list(map(lambda x: strategy_set[x], res_list))
        mem_cost = self.max_mem - mem_remain if mem_remain >= 0 else np.inf
        comm_cost = comm_cost * pp_deg

        # Write search result into history
        if self.search_history is not None and best_strategy_flag:
            self.search_history[key]={'comm_cost': comm_cost, 'res_list': res_list, 'mem_cost': mem_cost}
        return comm_cost, res_list, mem_remain, mem_cost, best_strategy_flag, from_history

    def _build_dp_and_run_multi_layer_type(self, pp_deg, bsz):
        # Look for results in search history
        history_results = []
        for i in range(pp_deg):
            key = (bsz, pp_deg, i)
            if self.search_history is not None and key in self.search_history.keys() and self.search_history[key]['mem_cost'] <= self.max_mem:
                history_results.append(self.search_history[key])
            else:
                history_results.append(None)

        strategy_set = list(filter(lambda s: s[0] == pp_deg, self.strategies_set))
        strategy_num = len(strategy_set)

        intra_layer_cost_list = []
        v_list = []
        for i in range(len(self.layer_num)):
            intra_layer_cost = [self.timecost_model(strategy, bsz, **self.timecost_model_args[i]).gen_result() for strategy in strategy_set]
            intra_layer_cost = np.array(intra_layer_cost, dtype=np.float64).reshape(1, -1).repeat(self.layer_num[i], axis=0)
            intra_layer_cost_list.append(intra_layer_cost)

            mem_cost_list = [self.memcost_model(strategy, bsz, **self.memcost_model_args[i]).get_memory_cost() for strategy in strategy_set]
            other_mem_cost = np.ceil(mem_cost_list[0]['other']).astype(int)
            v = [cost['enc_total'] for cost in mem_cost_list]
            v = np.ceil(np.array(v)).astype(np.int32)
            v = v.reshape(1, -1).repeat(self.layer_num[i], axis=0)
            v_list.append(v)
        
        intra_layer_cost = np.concatenate(intra_layer_cost_list, axis = 0)
        v = np.concatenate(v_list, axis = 0)
        min_cost_strategy_ids = np.argmin(intra_layer_cost, axis=1)

        # NEW VERSION: inter-layer timecost model
        inter_layer_cost = np.zeros((strategy_num, strategy_num))
        for i in range(strategy_num):
            for j in range(strategy_num):
                case1 = strategy_set[j][1] > strategy_set[i][1]
                case2 = False
                case3 = False
                if 'tp' in strategy_set[j][-1].keys() and 'tp' in strategy_set[i][-1].keys():
                    case2 = (strategy_set[j][1] == strategy_set[i][1] and strategy_set[j][-1]['tp'] != strategy_set[i][-1]['tp'])
                    world_size = strategy_set[i][1] * strategy_set[i][2]
                    case3 = (world_size == 8 and strategy_set[i][1] == 4 and strategy_set[j][1] == 2 \
                        and strategy_set[j][-1]['tp'] != strategy_set[i][-1]['tp'])
                if case1 or case2 or case3:
                     ratio = strategy_set[j][1]
                     activation = 2 * bsz / strategy_set[j][2]
                     inter_layer_cost[i, j] = (ratio - 1) * activation / ratio

        # find corresponding communication coefficient
        for i in range(strategy_num):
            for j in range(strategy_num):
                tp_size, dp_size = strategy_set[j][1], strategy_set[j][2]
                if tp_size == 1 or dp_size == 1:
                    coe = self.comm_coe_dict[pp_deg]['%d'%tp_size]
                else:
                    # In this case, strategy[-1]['tp'] represents tp_consecutive_flag
                    info = strategy_set[j][-1]
                    assert 'tp' in info.keys() and info['tp'] in [0, 1]
                    if info['tp']:
                        coe = self.comm_coe_dict[pp_deg]['%d_1'%tp_size]
                    else:
                        coe = self.comm_coe_dict[pp_deg]['%d_0'%tp_size]
                inter_layer_cost[i, j] = inter_layer_cost[i, j] * coe * 0.001

                # add a small bias to sort fsdp and dp
                strategy0, strategy1 = strategy_set[i], strategy_set[j]
                if i != j and np.array_equal(strategy0[:3], strategy1[:3]):
                    case1 = 'tp' not in strategy0[-1] and strategy0[-1]['fsdp']!=strategy1[-1]['fsdp']
                    case2 = 'tp' in strategy0[-1] and strategy0[-1]['tp']==strategy1[-1]['tp'] and strategy0[-1]['fsdp']!=strategy1[-1]['fsdp']
                    if (case1 or case2) and strategy0[-1]['fsdp']:
                        inter_layer_cost[i, j] = 1e-4
        
        inter_layer_cost = np.expand_dims(inter_layer_cost, axis=0).repeat(self.total_layer_num, axis=0)
        inter_layer_cost[0, :, :] = 0 # no inter-layer communication cost in first layer

        pp_stage_list = self.pp_stage_dict[pp_deg]
        start_layer = 0
        comm_cost_list, res_list_list, mem_remain_list, mem_cost_list = [], [], [], []
        best_strategy_flag, from_history = [False for i in range(pp_deg)], [False for i in range(pp_deg)]
        for i in range(pp_deg):
            # Apply history result
            if history_results[i] is not None:
                re = history_results[i]
                comm_cost, res_list, mem_remain, mem_cost = \
                    re['comm_cost'], re['res_list'], self.max_mem-re['mem_cost'], re['mem_cost']
                best_strategy_flag[i], from_history[i] = True, True
            else:
                if self.max_mem - other_mem_cost[i] <= 0:
                    return np.inf, None, -1, np.inf, False, False
                dp = DPAlg(self.max_mem - other_mem_cost[i], pp_stage_list[i], strategy_num)
                dp.set_v_and_cost(v[start_layer:start_layer+pp_stage_list[i]], 
                                    intra_layer_cost[start_layer:start_layer+pp_stage_list[i]], 
                                    inter_layer_cost[start_layer:start_layer+pp_stage_list[i]])
                comm_cost, res_list, mem_remain = dp.fit()
                best_strategy_flag[i] = res_list is not None and (np.array(res_list) == min_cost_strategy_ids[start_layer:start_layer+pp_stage_list[i]]).all()
                if res_list is not None:
                    res_list = list(map(lambda x: strategy_set[x], res_list))
                mem_cost = self.max_mem - mem_remain if mem_remain >= 0 else np.inf
                # Write search result into history
                if self.search_history is not None and best_strategy_flag[i]:
                    key = (bsz, pp_deg, i)
                    self.search_history[key]={'comm_cost': comm_cost, 'res_list': res_list, 'mem_cost': mem_cost}
            comm_cost_list.append(comm_cost)
            res_list_list.append(res_list)
            mem_remain_list.append(mem_remain)
            mem_cost_list.append(mem_cost)
            start_layer += pp_stage_list[i]
        return sum(comm_cost_list), res_list_list, mem_remain_list, mem_cost_list, best_strategy_flag, from_history

    def fit(self, bsz, print_=True):
        min_comm_cost = np.inf
        min_res_list = None
        min_pp_deg = -1
        min_mem_remain = -1
        min_mem_cost = -1

        for pp_deg in self.ppdeg_set:
            if print_:
                print(f'bsz={bsz}, pp_deg={pp_deg}:', flush=True)
            if bsz % (self.gpu_num//pp_deg):
                comm_cost, res_list, mem_remain, mem_cost, best_strategy_flag, from_history = np.inf, None, -1, np.inf, False, False
                if print_:
                    print('Best strategy:', best_strategy_flag, '\nFrom history:', from_history)
                    print(f'time cost: {comm_cost}, memory remaining: {mem_remain}, memory cost: {mem_cost}')
                continue
            if self.multi_layer_type:
                comm_cost, res_list, mem_remain, mem_cost, best_strategy_flag, from_history = self._build_dp_and_run_multi_layer_type(pp_deg, bsz)
            else:
                comm_cost, res_list, mem_remain, mem_cost, best_strategy_flag, from_history = self._build_dp_and_run(pp_deg, bsz)
            if print_:
                print('Best strategy:', best_strategy_flag, '\nFrom history:', from_history)
                print(f'time cost: {comm_cost}, memory remaining: {mem_remain}, memory cost: {mem_cost}')
            if min_comm_cost > comm_cost:
                min_res_list = res_list
                min_comm_cost = comm_cost
                min_pp_deg = pp_deg
                min_mem_remain = mem_remain
                min_mem_cost = mem_cost

        return min_comm_cost, min_res_list, min_pp_deg, min_mem_remain, min_mem_cost