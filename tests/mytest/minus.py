import numpy as np
import hetu as ht
from hetu import gpu_links as gpu_op

b=np.ones(10)
a=3*b
ctx=ht.gpu(0)
arr_a = ht.array(a,ctx=ctx)
arr_b = ht.array(b,ctx=ctx)
arr_c = ht.empty(arr_a.shape, ctx=ctx)

gpu_op.matrix_elementwise_minus(arr_a, arr_b, arr_c)

c=arr_c.asnumpy()

print(c)

