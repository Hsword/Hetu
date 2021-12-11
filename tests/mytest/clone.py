import numpy as np
import hetu as ht
from hetu import gpu_links as gpu_op

ctx=ht.gpu(0)
a=np.ones(10)
arr_a = ht.array(a,ctx=ctx)
arr_c = ht.empty(arr_a.shape, ctx=ctx)

gpu_op.clone(arr_a, arr_c)

c=arr_c.asnumpy()

print(c)

