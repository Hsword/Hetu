from .base import BaseLayer
import hetu as ht
import math
import numpy as np

def balance_loss(gates, mask, num_experts):
    me = ht.reduce_mean_op(gates, axes=0)
    ce = ht.reduce_mean_op(mask, axes=0)
    num_experts = ht.Variable('num_experts', value=np.array((num_experts,), dtype=np.float32), trainable=False)
    l_aux = ht.reducesumaxiszero_op(me * ce) * num_experts
    return l_aux

def alignment_loss(gates, top1_group, topk_indice, num_local_gpus):
    tmp = ht.sam_max_op(gates, top1_group, topk_indice, num_local_gpus)
    l_alignment = ht.reduce_sum_op(tmp, axes=0)
    l_alignment = ht.reduce_sum_op(l_alignment, axes=0)

    return l_alignment


def samgating(logits, k, capacity_factor: float, num_tokens: int, num_experts: int, embed_dim: int, num_local_gpus: int):
    """Implements SAMGating on logits."""
    # everything is in fp32 in this function
    gates = ht.softmax_op(logits)
    capacity = k * math.ceil((num_tokens / num_experts) * capacity_factor)
    
    group_sum = ht.sam_group_sum_op(gates, num_local_gpus)
    top1_group = ht.topk_idx_op(group_sum, topk=1)
    topk_indices = ht.group_topk_idx_op(gates, top1_group, topk=k, num_local_gpus=num_local_gpus)
    indices_s = []
    for i in range(k):
        indices_s.append(ht.split_op(topk_indices, axes=[1,], indices=[i,], splits=[k,]))
    topk_indice = indices_s[k-1]

    mask_topk = []
    for i in range(k):
        mask_topk.append(ht.array_reshape_op(ht.one_hot_op(indices_s[i], num_classes=num_experts), [-1, num_experts]))
    
    l_aux = balance_loss(gates, mask_topk[0], num_experts)
    l_alignment = alignment_loss(gates, top1_group, topk_indice, num_local_gpus)

    locations1 = ht.cumsum_with_bias_op(mask_topk[0], bias = -1, dim=0)

    location_s = [ht.reduce_sum_op(locations1 * mask_topk[0], axes=1)]



    acc_base = None
    for i in range(1, k):
        acc_base = ht.reduce_sum_op(mask_topk[k - 1], axes=0, keepdims=True) if acc_base is None else acc_base + ht.reduce_sum_op(mask_topk[k - 1], axes=0, keepdims=True)
        locations2 = ht.cumsum_with_bias_op(mask_topk[i], bias = -1, dim=0)
        locations2 += acc_base
        location_s.append(ht.reduce_sum_op(locations2 * mask_topk[i], axes=1))
        l_aux += balance_loss(gates, mask_topk[i], num_experts)

    tmp = ht.mul_op(gates, mask_topk[0])
    gates_s = [ht.reduce_sum_op(tmp, axes = 1)]
    for i in range(1, k):
        tmp = ht.mul_op(gates, mask_topk[i])
        gates_s.append(ht.reduce_sum_op(tmp, axes = 1))
        

    return l_aux, l_alignment, indices_s, location_s, gates_s, capacity
    
class SAMGate(BaseLayer):
    def __init__(self, embed_dim: int, num_tokens: int, num_experts: int, k: int = 1,\
                       capacity_factor: float = 1.0, eval_capacity_factor: float = 1.0,\
                       initializer=ht.init.GenXavierUniform(), name="TopK_Gate", num_local_gpus = 8):
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.top_k = k
        self.num_tokens = num_tokens
        self.capacity_factor = capacity_factor
        self.eval_capacity_factor = eval_capacity_factor
        self.initializer = initializer
        self.name = name
        self.num_local_gpus = num_local_gpus


    def __call__(self, x):
        weight_var = self.initializer(
                shape=(self.embed_dim, self.num_experts), name=self.name+'_linear_weight')
        x =  ht.matmul_op(x, weight_var)
        bias_var = ht.init.zeros(
                shape=(1, self.num_experts), name=self.name+'_linear_bias')
        bias_var = ht.broadcastto_op(bias_var, x)
        x = x + bias_var        
        return samgating(x, self.top_k, self.capacity_factor, self.num_tokens, self.num_experts, self.embed_dim, self.num_local_gpus)
