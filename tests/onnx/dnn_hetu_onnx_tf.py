import hetu as ht
from hetu import init
from hetu import onnx as ax

import onnxruntime as rt

import numpy as np

import onnx
from onnx_tf.backend import prepare

import tensorflow.compat.v1 as tf


import argparse
import six.moves.cPickle as pickle
import gzip
import os
import pdb
import ctypes
import time
batch_size = 2


# ctx=ht.gpu(0)
executor_ctx = ht.cpu(0)


def dnn(logs):

    print("Build DNN model...")
    logs.append('\n----------Hetu model to onnx to tensorflow(DNN)--------\n')
    logs.append('step 1: Building dnn models based on Hetu....')
    print(logs[-1])

    W1 = init.random_normal((784, 256), stddev=0.1, name='W1')
    b1 = init.random_normal((256,), stddev=0.1, name='b1')
    W2 = init.random_normal((256, 256), stddev=0.1, name='W2')
    b2 = init.random_normal((256,), stddev=0.1, name='b2')
    W3 = init.random_normal((256, 10), stddev=0.1, name='W3')
    b3 = init.random_normal((10,), stddev=0.1, name='b3')

    X = ht.Variable(name="X")

    z1 = ht.matmul_op(X, W1)+b1
    z2 = ht.relu_op(z1)

    z3 = ht.matmul_op(z2, W2)+b2
    z4 = ht.relu_op(z3)

    z5 = ht.matmul_op(z4, W3)+b3

    y = z5

    executor = ht.Executor(
        [y],
        ctx=executor_ctx)

    rand = np.random.RandomState(seed=123)
    X_val = rand.normal(scale=0.1, size=(batch_size, 784)).astype(np.float32)

    ath = executor.run(
        feed_dict={
            X: X_val})

    onnx_input_path = 'hetu_dnn_model.onnx'

    logs.append(
        'step 2: Loading Hetu model to onnx,filename is {}'.format(onnx_input_path))
    print(logs[-1])

    ax.hetu2onnx.export(executor, [X], [y], onnx_input_path)

    logs.append(
        'step 3: Validing onnx model between Hetu runtime and onnxruntime....')
    print(logs[-1])

    sess = rt.InferenceSession(onnx_input_path)
    input = sess.get_inputs()[0].name

    pre = sess.run(None, {input: X_val.astype(np.float32)})[0]

    np.testing.assert_allclose(ath[0].asnumpy(), pre, rtol=1e-2)

    logs.append('pass!')
    print(logs[-1])

    logs.append('step 4: Loading hetu_onnx model to tensorflow....')
    print(logs[-1])

    # convert onnx to tf
    onnx_model = onnx.load(onnx_input_path)  # load onnx model
    tf_exp = prepare(onnx_model)  # ,strict=False)  # prepare tf representation

    onnx_output = tf_exp.run(X_val)

    logs.append(
        'step 5: Validing onnx model between Hetu runtime and tensorflow runtime....')
    print(logs[-1])

    np.testing.assert_allclose(onnx_output[0], ath[0].asnumpy(), rtol=1e-2)
    logs.append('pass!')
    print(logs[-1])

    logs.append('\n----------Hetu model to onnx to tensorflow(DNN)--------end\n')
    print(logs[-1])


if __name__ == '__main__':
    logs = []
    dnn(logs)

    print('--------print all logs once!-------')
    for log in logs:
        print(log)
    print('--------logs end!--------')
