import numpy as np
import hetu as ht
from hetu import layers as htl
from tester import HetuTester
import argparse
import logging
           
def test_split():
    input_shape = (4,4)
    
    input_vals = np.random.normal(size=input_shape)
    
    input = ht.Variable(name='input', ctx=ht.gpu(0))
    
    split_op = ht.split_op(input, axes=[0, 0], indices = [2,3], splits = [4,4 ])
    
    gpu_executor = ht.Executor([split_op], ctx=ht.gpu(0))
    
    gpu_result = gpu_executor.run(feed_dict={input:input_vals})
    
    print(input_vals)
    for val in gpu_result:
        print(val.asnumpy())

def test_reduce_sum():
    input_shape = (4,4)
    input_vals = np.random.normal(size=input_shape)
    input = ht.Variable(name='input', ctx=ht.gpu(0))
    reduce_sum_op = ht.reduce_sum_op(input, axes=0)
    reduce_sum_op = ht.reduce_sum_op(reduce_sum_op, axes=0)
    gpu_executor = ht.Executor([reduce_sum_op], ctx=ht.gpu(0))
    gpu_result = gpu_executor.run(feed_dict={input:input_vals})
    print(input_vals)
    for val in gpu_result:
        print(val.asnumpy())

test_reduce_sum()
