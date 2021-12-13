import numpy as np
import hetu as ht
from hetu import gpu_links as gpu_op
from copy import deepcopy


ctx = ht.gpu(0)
x = np.ones(32,dtype=np.float).reshape(8,4)
arr_x = ht.array(x, ctx=ctx)
x_t = ht.empty((arr_x.shape[1], arr_x.shape[0]), ctx=ctx)
gpu_op.matrix_transpose(arr_x, x_t, perm=[1,0])
print(x_t.asnumpy())
arr_output_idx = ht.empty((4,3), ctx=ctx)
gpu_op.topk_idx(x_t, arr_output_idx, 3)

print(arr_output_idx.asnumpy())
