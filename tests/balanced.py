import numpy as np
import hetu as ht
from hetu import gpu_links as gpu_op
import sys

def balanced_assignment(scores, max_iterations=100):
    # scores [num_jobs * num_workers] 2-d ndarray
    # first, need to transpose scores
    ctx = scores.ctx
    scores_t = ht.empty((scores.shape[1], scores.shape[0]), ctx = ctx)
    gpu_op.matrix_transpose(scores, scores_t, perm = [1, 0])
    # find eps
    scores_numpy=scores.asnumpy()
    eps=(scores_numpy.max()-scores_numpy.min())/50;
    if eps < 1e-04:
        eps=1e-04
    # just follow the pseudo code
    num_workers, num_jobs = scores_t.shape
    jobs_per_worker = num_jobs//num_workers
    value = ht.empty(scores_t.shape, ctx=ctx)
    gpu_op.clone(scores_t, value)

    iterations = 0
    cost=np.zeros((1, num_jobs)).astype(np.float32)
    cost=ht.array(cost, ctx=ctx)
    
    jobs_with_bids = np.zeros(num_workers, dtype=bool)

    while not jobs_with_bids.all(): 
        # find top_index, top_value of value
        top_index_shape = [value.shape[0], jobs_per_worker+1]
        top_index = ht.empty(top_index_shape, ctx=ctx)
        gpu_op.topk_idx(value, top_index, jobs_per_worker+1)
        top_values = ht.empty(top_index_shape, ctx=ctx)
        gpu_op.topk_val(value, top_index, top_values, jobs_per_worker+1)
        # find bid_increments
        top_values_numpy = top_values.asnumpy()
        bid_increments = top_values_numpy[:,:-1]-top_values_numpy[:,-1:]+eps
        bid_increments = ht.array(bid_increments, ctx=ctx)
        # do scatter
        top_index_slice = top_index.asnumpy()[:,:-1]
        top_index_slice = ht.array(top_index_slice, ctx=ctx)
        bids = np.zeros((num_workers, num_jobs)).astype(np.float32)
        bids = ht.array(bids, ctx=ctx)
        gpu_op.scatter(bids, 1, top_index_slice, bid_increments) 

        if 0 < iterations < max_iterations:
            # follow pseudo code, use numpy index operation
            bids_numpy = bids.asnumpy()
            top_bidders_numpy=top_bidders.asnumpy()
            bids_numpy[top_bidders_numpy.astype(int), jobs_with_bids]=eps
            bids = ht.array(bids_numpy, ctx=ctx)
        
        # find top_1 
        top_bidders = ht.empty((1,bids.shape[1]), ctx=ctx)
        top_bids = ht.empty((1, bids.shape[1]), ctx=ctx)
        gpu_op.max(bids, top_bidders, top_bids, 0)
        top_bidders = top_bidders.asnumpy().flatten()
        top_bidders = ht.array(top_bidders, ctx=ctx)
        top_bids = top_bids.asnumpy().flatten()
        top_bids = ht.array(top_bids, ctx=ctx)
        
        # find top_bidders, top_bids
        top_bids_numpy = top_bids.asnumpy()
        jobs_with_bids = top_bids_numpy > 0
        top_bidders_numpy = top_bidders.asnumpy()
        top_bidders_numpy = top_bidders_numpy[jobs_with_bids]
        top_bidders = ht.array(top_bidders_numpy, ctx=ctx) 
        
        gpu_op.matrix_elementwise_add(cost, top_bids, cost)
        
        cost_broadcast=ht.empty(value.shape, ctx=ctx)
        gpu_op.broadcast_shape(cost, cost_broadcast)
        gpu_op.matrix_elementwise_minus(scores_t, cost_broadcast, value)
        
        if iterations < max_iterations:
            value_numpy = value.asnumpy() 
            top_bidders_numpy = top_bidders.asnumpy()
            value_numpy[top_bidders_numpy.astype(int), jobs_with_bids]=1e12
            value=ht.array(value_numpy, ctx=ctx)
        else:
            value_numpy = value.asnumpy()
            top_bidders_numpy = top_bidders.asnumpy()
            scores_numpy = scores_t.asnumpy()
            value_numpy[top_bidders_numpy.astype(int), jobs_with_bids]=scores_numpy[top_bidders_numpy.astype(int), jobs_with_bids]
            value=ht.array(value_numpy, ctx=ctx)
        iterations += 1

    top_index_slice = top_index.asnumpy()[:,:-1].reshape(-1)
    print(top_index_slice)

if __name__ == '__main__':
    np.random.seed(3)
    a=np.random.random([8192,8])
    ctx=ht.gpu(0)
    scores = ht.array(a, ctx=ctx)
    balanced_assignment(scores) 
