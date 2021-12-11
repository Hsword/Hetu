import numpy as np
import hetu as ht
from hetu import gpu_links as gpu_op

ctx = ht.gpu(0)
x=np.arange(10).reshape(1,10)
arr_x = ht.array(x, ctx=ctx)
arr_y = ht.empty((10,10), ctx=ctx)
gpu_op.broadcast_shape(arr_x, arr_y)
y = arr_y.asnumpy()

print(y)
