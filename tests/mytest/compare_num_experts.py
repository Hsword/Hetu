import numpy as np
import hetu as ht
from hetu import gpu_links as gpu_op
from copy import deepcopy
import torch
import time

ROW=8192
k=2
ctx = ht.gpu(0)
for i in range(1, 20):
    COL=i*10
    shape = (ROW, COL)
    x = np.random.uniform(0, 10, size=shape).astype(np.float32)
    arr_x = ht.array(x, ctx=ctx)
    output_shape=(ROW,k)
    arr_output_val = ht.empty(output_shape, ctx=ctx)
    arr_output_idx = ht.empty(output_shape, ctx=ctx)
    time_start=time.time()
    for i in range(20):
        gpu_op.topk(arr_x, arr_output_val, arr_output_idx, k)
    time_end=time.time()
    print("COL,"+str(COL)+",hetu,"+str(time_end-time_start))

    torch_x = torch.tensor(x, device='cuda:1')
    time_start=time.time()
    for i in range(20):
        torch.topk(torch_x, 2, dim=1)
    time_end=time.time()
    print("COL,"+str(COL)+",torch,"+str(time_end-time_start))
