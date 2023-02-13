from .base import BaseLayer
import hetu as ht
import math
import numpy as np

def hashgating(logits, capacity_factor: float, num_tokens: int, num_experts: int, embed_dim: int, indice:any):
    """Implements HashGating on logits."""
    """Currently Random Hash"""
    # round-up
    capacity = math.ceil((num_tokens / num_experts) * capacity_factor)
    indices_s = [indice]
    mask_topk = []
    mask_topk.append(ht.array_reshape_op(ht.one_hot_op(indices_s[0], num_classes=num_experts), [-1, num_experts]))
    locations1 = ht.cumsum_with_bias_op(mask_topk[0], bias = -1, dim=0)
    location_s = [ht.reduce_sum_op(locations1 * mask_topk[0], axes=1)] 

    return indices_s, location_s, capacity
    
class HashGate(BaseLayer):
    def __init__(self, embed_dim: int, num_tokens: int, num_experts: int, \
                       capacity_factor: float = 1.0, eval_capacity_factor: float = 1.0,\
                       name="Hash_Gate"):
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.num_tokens = num_tokens
        self.capacity_factor = capacity_factor
        self.eval_capacity_factor = eval_capacity_factor
        self.name = name

    def __call__(self, x, indice):
        return hashgating(x, self.capacity_factor, self.num_tokens, self.num_experts, self.embed_dim, indice)
