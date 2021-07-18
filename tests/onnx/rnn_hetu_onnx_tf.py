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

diminput = 28
dimhidden = 128
dimoutput = 10
nsteps = 28

# ctx=ht.gpu(0)
executor_ctx = ht.cpu(0)


def rnn(logs):

    print("Build RNN model...")
    logs.append('\n----------Hetu model to onnx to tensorflow(RNN)--------\n')
    logs.append('step 1: Building rnn(nsteps=28) models based on Hetu....')
    print(logs[-1])

    W1 = init.random_normal((diminput, dimhidden), stddev=0.1, name='W1')
    b1 = init.random_normal((dimhidden,), stddev=0.1, name='b1')
    W2 = init.random_normal(
        (dimhidden+dimhidden, dimhidden), stddev=0.1, name='W2')
    b2 = init.random_normal((dimhidden,), stddev=0.1, name='b2')
    W3 = init.random_normal((dimhidden, dimoutput), stddev=0.1, name='W3')
    b3 = init.random_normal((dimoutput,), stddev=0.1, name='b3')
    last_state = init.zeros(shape=(batch_size, dimhidden), name='last_state')

    #
    # last_state = ht.Variable(value=np.zeros((batch_size, dimhidden)).astype(np.float32), name='initial_state', trainable=False)
    # #last_state = np.zeros((batch_size, dimhidden)).astype(np.float32)

    X = ht.Variable(name="X")

    for i in range(nsteps):
        cur_x = ht.slice_op(X, (0, i * diminput), (-1, diminput))
        h = ht.matmul_op(cur_x, W1) + b1

        # if i == 0:
        #     last_state = ht.broadcastto_op(last_state, h)
        s = ht.concat_op(h, last_state, axis=1)
        s = ht.matmul_op(s, W2) + b2
        last_state = ht.relu_op(s)

    final_state = last_state
    z1 = ht.matmul_op(final_state, W3) + b3
    y = z1

    executor = ht.Executor(
        [y],
        ctx=executor_ctx)

    rand = np.random.RandomState(seed=123)
    X_val = rand.normal(scale=0.1, size=(batch_size, 784)).astype(np.float32)

    ath = executor.run(
        feed_dict={
            X: X_val})

    onnx_input_path = 'hetu_rnn_model.onnx'

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

    logs.append('\n----------Hetu model to onnx to tensorflow(RNN)--------end\n')
    print(logs[-1])


if __name__ == '__main__':
    logs = []
    rnn(logs)

    print('--------print all logs once!-------')
    for log in logs:
        print(log)
    print('--------logs end!--------')
