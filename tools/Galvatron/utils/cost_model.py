import numpy as np

class MemoryCostModel:
    def __init__(self,
            strategy,
            global_batch_size = 8,
            parameter_size = 48,
            tp_activation_per_bsz_dict = {1:85, 2:47, 4:28, 8:18.5},
            other_model_states = 640, 
            other_activation_per_bsz = 320,
            pytorch_context_mem = 1024):
        self.strategy = strategy
        self.pp_size = self.strategy[0]
        self.tp_size = self.strategy[1]
        self.dp_size = self.strategy[2]
        self.parameter_size = parameter_size/self.tp_size
        self.model_states_size = 4 * self.parameter_size
        if 'fsdp' in self.strategy[-1].keys() and self.strategy[-1]['fsdp']:
            # fsdp_model_states memory is slightly larger than dp_model_states/dp_size
            # we add a small bias to ensure the predicted fsdp memory NOT smaller than real value
            # Actually, this bias barely affect search result.
            self.model_states_size  *= (1/self.dp_size + 0.025)
        self.bsz = global_batch_size/self.dp_size
        self.activation_size = tp_activation_per_bsz_dict[self.tp_size] * self.bsz
        self.total = self.model_states_size + self.activation_size
        self.other_memcost = other_model_states + other_activation_per_bsz * (global_batch_size/self.tp_size/self.dp_size)
        self.other_memcost += pytorch_context_mem

    def get_memory_cost(self):
        result = dict()
        result['parameter'] = self.parameter_size
        result['model_states'] = self.model_states_size
        result['activation'] = self.activation_size
        result['enc_total'] = self.total
        result['other'] = self.other_memcost
        return result

class TimeCostModel_with_overlap:
    def __init__(self,
            strategy,
            global_batch_size,
            parameter_size = 48,
            microbatch=True,
            optimal_chunk_func = None,
            sequence_length=512,
            hidden_size=1024,
            forward_computation_time=35 / 24,
            bct_fct_coe=2,
            extra_overhead=0,
            comm_coe_dict={},
            dp_overlap_coe=1.3,
            bct_overlap_coe=1.3,
            layer_type='enc'):
        self.s = strategy[:3]
        self.sl = sequence_length
        self.hs = hidden_size
        self.microbatch = microbatch
        self.pp_size = self.s[0]
        self.tp_size = self.s[1]
        self.dp_size = self.s[2]
        self.comm_coe_dict = comm_coe_dict[self.pp_size]
        if self.tp_size == 1 or self.dp_size == 1:
            self.dc = self.comm_coe_dict['%d'%self.dp_size]
            self.tc = self.comm_coe_dict['%d'%self.tp_size]
        else:
            # In this case, strategy[-1]['tp'] represents tp_consecutive_flag
            info = strategy[-1]
            assert 'tp' in info.keys() and info['tp'] in [0, 1]
            tp_consecutive_flag = info['tp']
            if tp_consecutive_flag:
                self.dc = self.comm_coe_dict['%d_0'%self.dp_size]
                self.tc = self.comm_coe_dict['%d_1'%self.tp_size]
            else:
                self.dc = self.comm_coe_dict['%d_1'%self.dp_size]
                self.tc = self.comm_coe_dict['%d_0'%self.tp_size]
        self.fsdp = False
        if 'fsdp' in strategy[-1].keys() and strategy[-1]['fsdp']:
            self.fsdp = True
        self.dp_overlap_coe = dp_overlap_coe
        self.dc_overlap = self.dc*dp_overlap_coe
        self.ps = parameter_size/self.tp_size
        self.bs = global_batch_size/self.dp_size 
        self.layer_type = layer_type
        assert(layer_type in ['enc', 'dec'])
        if microbatch:
            self.optimal_microbatch = optimal_chunk_func(self.bs, self.s)
            # print(self.optimal_microbatch)

        # Dummy layer_num, can be any multiple of 8.
        # We estimate the time cost of single layer by averaging the time of whole model to deal with pipeline parallel
        self.layer_num = 24 

        # forward & backward computation time of whole model (depending on dummy layer_num)
        self.fct = forward_computation_time * self.bs / self.tp_size * self.layer_num 
        self.bct = self.fct * bct_fct_coe
        self.bct_overlap_coe = bct_overlap_coe
        self.bct_overlap = self.bct*bct_overlap_coe
        self.eo = extra_overhead

        # dp & tp message size of whole model (depending on dummy layer_num)
        self.dp_message_size = (2*(self.dp_size-1)/self.dp_size*self.ps) * self.layer_num
        tp_comm_times = 4 if layer_type=='enc' else 6
        self.tp_message_size = 2*(self.tp_size-1)/self.tp_size*(self.bs*self.sl*self.hs*tp_comm_times*4/1024/1024) * self.layer_num

    def bct_dp_overlap(self, dp_message_size, bct):
        dp_overlap_time = dp_message_size * self.dc_overlap
        bct_overlap_time = bct * self.bct_overlap_coe
        if dp_overlap_time > bct_overlap_time:
            overlap_part = bct_overlap_time
            rest_part = (dp_message_size - bct_overlap_time / self.dc_overlap) * self.dc
            rest_dp_flag = True
        elif dp_overlap_time < bct_overlap_time:
            overlap_part = dp_overlap_time
            rest_part = (bct - dp_overlap_time / self.bct_overlap_coe) 
            rest_dp_flag = False
        else:
            overlap_part = bct_overlap_time
            rest_part = 0
            rest_dp_flag = False
        return overlap_part, rest_part, rest_dp_flag

    def pipe_with_microbatch(self, computation_overhead, communication_overhead):
        result = computation_overhead*(self.pp_size+self.optimal_microbatch-1)/(self.pp_size*self.optimal_microbatch)+communication_overhead
        return result

    def gen_result(self):
        if np.array_equal(self.s, [1,1,8]) == True:
            overlap_part, rest_part, _ = self.bct_dp_overlap(self.dp_message_size, self.bct)
            result = self.fct + overlap_part + rest_part + self.eo
        elif np.array_equal(self.s, [1,2,4]) == True:
            if self.layer_type == 'enc':
                overlap_part, rest_part, _ = self.bct_dp_overlap(self.dp_message_size, self.bct)
                result = self.fct + overlap_part + rest_part + self.tp_message_size*self.tc + self.eo
            elif self.layer_type == 'dec':
                overlap_part, rest_part, _ = self.bct_dp_overlap(self.dp_message_size, self.bct*2/3)
                result = self.fct + 1/3*self.bct + overlap_part + rest_part +self.tp_message_size*self.tc+self.eo
        elif np.array_equal(self.s, [1,4,2]) == True:
            if self.layer_type == 'enc':
                overlap_part, rest_part, _ = self.bct_dp_overlap(self.dp_message_size, self.bct*1/2)
                result = self.fct + 1/2*self.bct + overlap_part + rest_part + self.tp_message_size*self.tc + self.eo
            elif self.layer_type == 'dec':
                overlap_part, rest_part, _ = self.bct_dp_overlap(self.dp_message_size, self.bct*2/3)
                result = self.fct + 1/3*self.bct + overlap_part + rest_part + self.tp_message_size*self.tc + self.eo
        elif np.array_equal(self.s, [1,8,1]) == True:
            result = self.fct + self.bct + self.tp_message_size*self.tc
        elif np.array_equal(self.s, [2,1,4]) == True:
            bct_per_stage = self.bct / self.pp_size
            dp_message_size_per_stage = self.dp_message_size / self.pp_size
            overlap_part_per_stage, rest_part_per_stage, rest_dp_flag = self.bct_dp_overlap(dp_message_size_per_stage, bct_per_stage)
            if rest_dp_flag and not self.fsdp:
                overall_overhead = self.fct + overlap_part_per_stage * self.pp_size + rest_part_per_stage + self.eo
            else:
                overall_overhead = self.fct + (overlap_part_per_stage + rest_part_per_stage) * self.pp_size + self.eo
            if self.microbatch == False:
                result = overall_overhead
            else:
                computation_overhead = self.fct + self.bct
                communication_overhead = overall_overhead-computation_overhead
                result = self.pipe_with_microbatch(computation_overhead, communication_overhead)
        elif np.array_equal(self.s, [2,2,2]) == True:
            overlap_part, rest_part, _ = self.bct_dp_overlap(self.dp_message_size, self.bct*1/2)
            overall_overhead = self.fct + overlap_part + rest_part + self.bct*1/2 + self.tp_message_size*self.tc + self.eo
            if self.microbatch == False:
                result = overall_overhead
            else:
                computation_overhead = self.fct + self.bct + self.tp_message_size*self.tc
                communication_overhead = overall_overhead-computation_overhead
                result = self.pipe_with_microbatch(computation_overhead, communication_overhead)
        elif np.array_equal(self.s, [2,4,1]) == True:
            if self.microbatch == False:
                result = self.fct + self.bct + self.tp_message_size*self.tc
            else:
                overall_overhead = self.fct + self.bct + self.tp_message_size*self.tc
                result = self.pipe_with_microbatch(overall_overhead, 0)
        elif np.array_equal(self.s, [4,1,2]) == True:
            # new version
            bct_per_stage = self.bct / self.pp_size
            dp_message_size_per_stage = self.dp_message_size / self.pp_size
            overlap_part_per_stage, rest_part_per_stage, rest_dp_flag = self.bct_dp_overlap(dp_message_size_per_stage, bct_per_stage)
            if rest_dp_flag and not self.fsdp:
                overall_overhead = self.fct + overlap_part_per_stage * self.pp_size + rest_part_per_stage + self.eo
            else:
                overall_overhead = self.fct + (overlap_part_per_stage + rest_part_per_stage) * self.pp_size + self.eo
            if self.microbatch == False:
                result = overall_overhead
            else:
                computation_overhead = self.fct + self.bct
                communication_overhead = overall_overhead-computation_overhead
                result = self.pipe_with_microbatch(computation_overhead, communication_overhead)
        elif np.array_equal(self.s, [4,2,1]) == True:
            if self.microbatch == False:
                result = self.fct + self.bct + self.tp_message_size*self.tc
            else:
                overall_overhead = self.fct + self.bct + self.tp_message_size*self.tc
                result = self.pipe_with_microbatch(overall_overhead, 0)
        elif np.array_equal(self.s, [8,1,1]) == True:
            if self.microbatch == False:
                result = self.fct + self.bct
            else:
                overall_overhead = self.fct + self.bct
                result = self.pipe_with_microbatch(overall_overhead, 0)
        if self.fsdp:
            result = result + self.dp_message_size * 0.5 * self.dc
        coe = 0.0011
        result = result*coe
        result = result / self.layer_num
        return result