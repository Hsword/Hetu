import numpy as np
import hetu as ht
from hetu import gpu_links as gpu_op
from copy import deepcopy


ctx = ht.gpu(0)
x = np.arange(32,dtype=np.float32).reshape(8,4)
arr_x = ht.array(x, ctx=ctx)
idx=ht.empty((1,arr_x.shape[1]), ctx=ctx)
val=ht.empty((1,arr_x.shape[1]), ctx=ctx)

gpu_op.max(arr_x, idx, val,  0)
print(arr_x.asnumpy())
print(idx.asnumpy())
print(val.asnumpy())
