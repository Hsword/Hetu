import numpy as np
import hetu as ht
from hetu import gpu_links as gpu_op
from copy import deepcopy
import torch

ctx = ht.gpu(0)

tgt = np.zeros((4, 20)).astype(np.float32)
arr_tgt = ht.array(tgt, ctx=ctx)
print(tgt)

src = np.array([[20,16,12,8,4],[20,16,12,8,4],[20,16,12,8,4],[20,16,12,8,4],[20,16,12,8,4]]).astype(np.float32)
arr_src = ht.array(src, ctx=ctx)
print(src)
index = np.array([[19,18,17,16,15],[19,18,17,16,15],[19,18,17,16,15],[19,18,17,16,15],[19,18,17,16,15]]).astype(np.float32)
arr_index = ht.array(index, ctx=ctx)
print(index)
gpu_op.scatter(arr_tgt, 1,  arr_index, arr_src)

output_tgt = arr_tgt.asnumpy()
print(output_tgt)

