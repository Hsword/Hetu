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


def cnn(logs):

    print("Build CNN model...")
    logs.append('\n----------Hetu model to onnx to tensorflow(CNN)--------\n')
    logs.append('step 1: Building cnn models based on Hetu....')
    print(logs[-1])

    W1 = init.random_normal((32, 1, 5, 5), stddev=0.1, name='W1')
    W2 = init.random_normal((64, 32, 5, 5), stddev=0.1, name='W2')
    W3 = init.random_normal((7*7*64, 10), stddev=0.1, name='W3')
    b3 = init.random_normal((10,), stddev=0.1, name='b3')

    X = ht.Variable(name="X")

    z1 = ht.conv2d_op(X, W1, padding=2, stride=1)
    z2 = ht.relu_op(z1)
    z3 = ht.avg_pool2d_op(z2, kernel_H=2, kernel_W=2, padding=0, stride=2)

    z4 = ht.conv2d_op(z3, W2, padding=2, stride=1)
    z5 = ht.relu_op(z4)
    z6 = ht.avg_pool2d_op(z5, kernel_H=2, kernel_W=2, padding=0, stride=2)

    z6_flat = ht.array_reshape_op(z6, (-1, 7 * 7 * 64))
    y = ht.matmul_op(z6_flat, W3)+b3

    executor = ht.Executor(
        [y],
        ctx=executor_ctx)

    rand = np.random.RandomState(seed=123)
    X_val = rand.normal(scale=0.1, size=(
        batch_size, 1, 28, 28)).astype(np.float32)

    ath = executor.run(
        feed_dict={
            X: X_val})

    onnx_input_path = 'hetu_cnn_model.onnx'

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

    logs.append('\n----------Hetu model to onnx to tensorflow(CNN)--------end\n')
    print(logs[-1])


def onnx2tf(onnx_input_path):
    onnx_model = onnx.load(onnx_input_path)  # load onnx model
    tf_exp = prepare(onnx_model)  # ,strict=False)  # prepare tf representation
    rand = np.random.RandomState(seed=123)
    X_val = rand.normal(scale=0.1, size=(
        batch_size, 1, 28, 28)).astype(np.float32)

    onnx_output = tf_exp.run(X_val)

    logs.append('   Convert pass!')
    print(logs[-1])
    logs.append(
        'step 5: validing onnx model between Hetu runtime and onnxruntime....')
    print(logs[-1])
    sess = rt.InferenceSession(onnx_input_path)
    input = sess.get_inputs()[0].name

    pre = sess.run(None, {input: X_val.astype(np.float32)})[0]

    np.testing.assert_allclose(onnx_output[0], pre, rtol=1e-2)
    print('pass3')

    # pb_output_path = 'model.pb'
    # tf_exp.export_graph(pb_output_path)  # export the model
    #
    # with tf.Graph().as_default():
    #     output_graph_def = tf.GraphDef()
    #
    #     with open(pb_output_path, "rb") as f:
    #         output_graph_def.ParseFromString(f.read())
    #         tf.import_graph_def(output_graph_def, name="")
    #
    #     with tf.Session() as sess:
    #         tf.global_variables_initializer().run()
    #         inp = sess.graph.get_tensor_by_name('actual_input_1:0')   #
    #         # out0 = sess.graph.get_tensor_by_name('output1:0')
    #         # out1 = sess.graph.get_tensor_by_name('73:0')
    #         # out2 = sess.graph.get_tensor_by_name('74:0')
    #         #
    #         # img = np.load('random.npy')
    #         # # img = img.reshape([1, 3, 300, 300])
    #         # pre_num = sess.run([out0, out1, out2], feed_dict={inp: img})
    #         # print(pre_num)


if __name__ == '__main__':
    logs = []
    cnn(logs)

    print('--------print all logs once!-------')
    for log in logs:
        print(log)
    print('--------logs end!--------')
