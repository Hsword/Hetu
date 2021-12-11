"""
Naive gate
"""
from .base_gate import BaseGate

import hetu as ht

class NaiveGate(BaseGate):
    r"""
    A naive gate implementation that defines the standard behavior of a gate that determines which experts the tokens are going to. 
    """
    def __init__(self, d_model, num_expert, world_size, topk=2, initializer=ht.init.GenXavierUniform, name="NaiveGate"):
        super.__init__(num_expert, world_size)
        self.topk = topk
        self.initializer = initializer
        self.d_model = d_model
        self.name = name

    def __call__(self, x, return_all_score = False):
        weight_var = self.initializer(
                shape=(self.d_model, self.tot_expert), name=self.name+'_linear_weight')
        x =  ht.matmul_op(x, weight_var)
        bias_var = ht.init.zeros(
                shape=(1, self.tot_expert), name=self.name+'_linear_bias')
        bias_var = ht.broadcastto_op(bias_var, x) # ?
        x = x + bias_var # ?
        topk_idx = ht.topk_idx_op(x, self.topk)
        topk_val = ht.topk_idx_op(x, topk_idx, self.topk)
        gate_score = ht.softmax_op(topk_val)     
        if return_all_score:
            return topk_idx, topk_val, gate_score
        else:
            return topk_idx, topk_val



