import numpy as np
import hetu as ht
from hetu import gpu_links as gpu_op
from copy import deepcopy

shape = (4321, 1234)
ctx = ht.gpu(0)
x = np.random.uniform(-1, 1, shape).astype(np.float32)
arr_x = ht.array(x, ctx=ctx)
arr_y = ht.empty((shape[1], shape[0]), ctx=ctx)
gpu_op.matrix_transpose(arr_x, arr_y, perm=[1, 0])
y = arr_y.asnumpy()

print(x)

print(y)
