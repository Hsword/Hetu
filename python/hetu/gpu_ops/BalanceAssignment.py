from __future__ import absolute_import
import numpy as np
from .Node import Op
from copy import deepcopy
from hetu import gpu_links 
import hetu as ht

def balanced_assignment_cpu(scores, max_iterations):
    pass
    
def balanced_assignment_gpu(scores, output, max_iterations, stream_handle):
    ctx = scores.ctx
    scores_t=ht.empty((scores.shape[1], scores.shape[0]), ctx=ctx)
    gpu_links.matrix_transpose(scores, scores_t, perm=[1, 0])
    scores_numpy = scores.asnumpy()
    eps=(scores_numpy.max()-scores_numpy.min())/50
    if eps < 1e04:
        eps = 1e-4
    num_workers, num_jobs = scores_t.shape
    jobs_per_worker = num_jobs//num_workers
    value = ht.empty(scores_t.shape, ctx=ctx)
    gpu_links.clone(scores_t, value)
    iterations=0
    cost=np.zeros((1, num_jobs)).astype(np.float32)
    cost=ht.array(cost, ctx=ctx)
    jobs_with_bids=np.zeros(num_workers, dtype=bool)
    while not jobs_with_bids.all():
        top_index_shape=[value.shape[0], jobs_per_worker+1]
        top_index=ht.empty(top_index_shape, ctx=ctx)
        gpu_links.topk_idx(value, top_index, jobs_per_worker+1)
        top_values=ht.empty(top_index_shape, ctx=ctx)
        gpu_links.topk_val(value, top_index, top_values, jobs_per_worker+1)
        top_values_numpy=top_values.asnumpy()
        bid_increments=top_values_numpy[:,:-1]-top_values_numpy[:,-1:]+eps
        bid_increments=ht.array(bid_increments, ctx=ctx)
        top_index_slice=top_index.asnumpy()[:,:-1]
        top_index_slice=ht.array(top_index_slice, ctx=ctx)
        bids=np.zeros((num_workers, num_jobs)).astype(np.float32)
        bids=ht.array(bids, ctx=ctx)
        gpu_links.scatter(bids, 1, top_index_slice, bid_increments)

        if 0<iterations<max_iterations:
            bids_numpy=bids.asnumpy()
            top_bidders_numpy=top_bidders.asnumpy()
            bids_numpy[top_bidders_numpy.astype(int), jobs_with_bids]=eps
            bids=ht.array(bids_numpy, ctx=ctx)
        
        top_bidders=ht.empty((1, bids.shape[1]), ctx=ctx)
        top_bids=ht.empty((1, bids.shape[1]), ctx=ctx)
        gpu_links.max(bids, top_bidders, top_bids, 0)
        top_bidders=top_bidders.asnumpy().flatten()
        top_bidders=ht.array(top_bidders, ctx=ctx)
        top_bids=top_bids.asnumpy().flatten()
        top_bids=ht.array(top_bids, ctx=ctx)

        top_bids_numpy=top_bids.asnumpy()
        jobs_with_bids=top_bids_numpy>0
        top_bidders_numpy=top_bidders.asnumpy()
        top_bidders_numpy=top_bidders_numpy[jobs_with_bids]
        top_bidders=ht.array(top_bidders_numpy, ctx=ctx)
        gpu_links.matrix_elementwise_add(cost, top_bids, cost)
        cost_broadcast=ht.empty(value.shape, ctx=ctx)
        gpu_links.broadcast_shape(cost, cost_broadcast)
        gpu_links.matrix_elementwise_minus(scores_t, cost_broadcast, value)
        
        if iterations<max_iterations:
            value_numpy=value.asnumpy()
            top_bidders_numpy=top_bidders.asnumpy()
            value_numpy[top_bidders_numpy.astype(int), jobs_with_bids]=1e12
            value=ht.array(value_numpy, ctx=ctx)
        else:
            value_numpy=value.asnumpy()
            top_bidders_numpy=top_bidders.asnumpy()
            scores_numpy=scores_t.asnumpy()
            value_numpy[top_bidders_numpy.astype(int), jobs_with_bids]=scores_numpy[top_bidders_numpy.astype(int), jobs_with_bids]
            value=ht.array(value_numpy, ctx=ctx)
        iterations+=1

    top_index_slice=top_index.asnumpy()[:,:-1].reshape(-1)
    top_index_slice=ht.array(top_index_slice, ctx=ctx)
    gpu_links.clone(top_index_slice, output)
    




class BalanceAssignmentOp(Op):
    def __init__(self, node_A, max_iterations=100, ctx=None):
        super().__init__(BalanceAssignmentOp, [node_A], ctx)
        self.max_iterations = max_iterations

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            scores = input_vals[0].asnumpy()
            balanced_assignment_cpu(scores, self.max_iterations)
        else:
            scores = input_vals[0]
            balanced_assignment_gpu(scores, output_val, self.max_iterations, stream_handle)

    def gradient(self, output_grad):
        return None

    def infer_shape(self, input_shapes):
        assert len(input_shapes)==1
        return (input_shapes[0][0],)
           

def balance_assignment_op(node, ctx=None):
    return BalanceAssignmentOp(node,ctx=ctx)
