from .base import BaseLayer
import hetu as ht
import math
import numpy as np

def generate_orthogonal(input_shape, gain=0.1):
    rows = input_shape[0]
    cols = 1
    for i in range(1, len(input_shape)):
        cols *= input_shape[i]

    flattened = np.random.normal(0,1,(rows, cols))
    if rows < cols:
        flattened = flattened.T
    q, r = np.linalg.qr(flattened)
    d = np.diag(r, 0)
    ph = [1 if i>0 else -1 for i in d]
    q *= ph
    if rows < cols:
        q = q.T
    val = q*gain
    return val


class BalanceAssignmentGate(BaseLayer):
    def __init__(self, embed_dim: int, num_tokens: int, num_experts: int, \
                       capacity_factor: float = 1.0, eval_capacity_factor: float = 1.0,\
                       initializer=ht.init.GenXavierUniform(), name="BalanceAssignment_Gate", device_id=None):
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.num_tokens = num_tokens
        self.capacity_factor = capacity_factor
        self.eval_capacity_factor = eval_capacity_factor
        self.initializer = initializer
        self.name = name
        self.device_id=device_id
        self.expert_centroids = ht.Variable(value=generate_orthogonal((self.num_experts, self.embed_dim)), name='centroids', trainable=False)
    
    def __call__(self, x):
        expert_centroids_trans = ht.transpose_op(self.expert_centroids)
        temp = ht.matmul_op(x, expert_centroids_trans)
        indice = ht.balance_assignment_op(temp)
        centroid = ht.split_op(self.expert_centroids, axes=[0,], indices=[self.device_id,],splits=[self.num_experts,])
        centroid = ht.array_reshape_op(centroid, [-1,1])

        return indice, centroid
