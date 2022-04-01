import numpy as np
import hetu as ht
from hetu import gpu_links as gpu_op
import sys

max_iterations=100
scores = np.arange(80).reshape(20,4)

ctx=ht.gpu(0)
scores = ht.array(scores, ctx=ctx)

ctx = scores.ctx
scores_t = ht.empty((scores.shape[1], scores.shape[0]), ctx = ctx)
gpu_op.matrix_transpose(scores, scores_t, perm = [1, 0])
eps = 0.0001 # later

num_workers, num_jobs = scores_t.shape
jobs_per_worker = num_jobs//num_workers
value = ht.empty(scores_t.shape, ctx=ctx)
gpu_op.clone(scores_t, value)
iterations = 0
cost=np.zeros((1, num_jobs)).astype(np.float32)
cost=ht.array(cost, ctx=ctx)
jobs_with_bids = np.zeros(num_workers, dtype=bool)

top_index_shape = [value.shape[0], jobs_per_worker+1]
top_index = ht.empty(top_index_shape, ctx=ctx)
gpu_op.topk_idx(value, top_index, jobs_per_worker+1)
top_values = ht.empty(top_index_shape, ctx=ctx)
gpu_op.topk_val(value, top_index, top_values, jobs_per_worker+1)
top_values_numpy = top_values.asnumpy()
bid_increments = top_values_numpy[:,:-1]-top_values_numpy[:,-1:]+eps
bid_increments = ht.array(bid_increments, ctx=ctx)
top_index_slice = top_index.asnumpy()[:,:-1]
top_index_slice = ht.array(top_index_slice, ctx=ctx)
bids = np.zeros((num_workers, num_jobs)).astype(np.float32)
bids = ht.array(bids, ctx=ctx)
gpu_op.scatter(bids, 1, top_index_slice, bid_increments)
if 0 < iterations < max_iterations:
#bids[top_bidders, jobs_with_bids] = eps # ???
    bids_numpy = bids.asnumpy()
    top_bidders_numpy=top_bidders.asnumpy()
    bids_numpy[top_bidders_numpy, jobs_with_bids]=eps
    bids = ht.array(bids_numpy, ctx=ctx)
bids_t = ht.empty((bids.shape[1], bids.shape[0]), ctx=ctx)
gpu_op.matrix_transpose(bids, bids_t, perm=[1,0])
top_bidders_t = ht.empty((bids_t.shape[0], 1), ctx=ctx)
gpu_op.topk_idx(bids_t, top_bidders_t, 1)
top_bids_t = ht.empty((bids_t.shape[0], 1), ctx=ctx)
gpu_op.topk_val(bids_t, top_bidders_t, top_bids_t, 1)
top_bidders = top_bidders_t.asnumpy().flatten()
top_bidders = ht.array(top_bidders, ctx=ctx)
top_bids = top_bids_t.asnumpy().flatten()
top_bids = ht.array(top_bids, ctx=ctx)
top_bids_numpy = top_bids.asnumpy()
jobs_with_bids = top_bids_numpy > 0
top_bidders_numpy = top_bidders.asnumpy()
top_bidders_numpy = top_bidders_numpy[jobs_with_bids]
top_bidders = ht.array(top_bidders_numpy, ctx=ctx)
gpu_op.matrix_elementwise_add(cost, top_bids, cost)
    # scores_t [num_workers*num_jobs], cost [1*num_jobs]
cost_broadcast=ht.empty(value.shape, ctx=ctx)
gpu_op.broadcast_shape(cost, cost_broadcast)
gpu_op.matrix_elementwise_minus(scores_t, cost_broadcast, value)
if iterations < max_iterations:
#value[top_bidders, job_with_bids] = inf # ???
    value_numpy = value.asnumpy()
    top_bidders_numpy = top_bidders.asnumpy()
    value_numpy[top_bidders_numpy.astype(int), jobs_with_bids]=sys.float_info.max
    value=ht.array(value_numpy, ctx=ctx)
else:
#value[top_bidders, jobs_with_bids] = scores[top_bidders, jobs_with_bids]
    value_numpy = value.asnumpy()
    top_bidders_numpy = top_bidders.asnumpy()
    scores_numpy = scores.asnumpy()
    value_numpy[top_bidders_numpy.astype(int), jobs_with_bids]=scores_numpy[top_bidders_numpy, jobs_with_bids]
    value=ht.array(value_numpy, ctx=ctx)
iterations += 1

top_index_slice = top_index.asnumpy()[:,:-1].reshape(-1)
print(top_index_slice)
