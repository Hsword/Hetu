import hetu as ht
from hetu import init
from hetu import onnx as hx

import onnxruntime as rt

import numpy as np


import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

import tf2onnx


import argparse
import six.moves.cPickle as pickle
import gzip
import os
import pdb
import ctypes
import time
batch_size = 128


# ctx=ht.gpu(0)
ctx = ht.cpu(0)


def cnn(executor_ctx=None, num_epochs=10, print_loss_val_each_epoch=False):

    print("Build CNN model...")

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

    hx.hetu2onnx.export(executor, [X], [y], 'ath.onnx')
    #
    #
    sess = rt.InferenceSession("ath.onnx")
    input = sess.get_inputs()[0].name

    pre = sess.run(None, {input: X_val.astype(np.float32)})[0]

    np.testing.assert_allclose(ath[0].asnumpy(), pre, rtol=1e-2)


def tf_model(logs, model_name='tf_cnn_model.onnx',):
    logs.append('Building cnn models based on tensorflow....')
    print(logs[-1])
    rand = np.random.RandomState(seed=123)
    X_val = rand.normal(scale=0.1, size=(20, 784)).astype(np.float32)

    with tf.Session() as sess:

        x = tf.placeholder(dtype=tf.float32, shape=(None, 784,), name='input')
        z1 = tf.reshape(x, [-1, 28, 28, 1])

        weight1 = tf.Variable(np.random.normal(scale=0.1, size=(
            32, 1, 5, 5)).transpose([2, 3, 1, 0]).astype(np.float32))
        z2 = tf.nn.conv2d(z1, weight1, padding='SAME', strides=[1, 1, 1, 1])
        z3 = tf.nn.relu(z2)
        z4 = tf.nn.avg_pool(
            z3, ksize=[1, 2, 2, 1], padding='VALID', strides=[1, 2, 2, 1])

        weight2 = tf.Variable(np.random.normal(scale=0.1, size=(
            64, 32, 5, 5)).transpose([2, 3, 1, 0]).astype(np.float32))
        z5 = tf.nn.conv2d(z4, weight2, padding='SAME', strides=[1, 1, 1, 1])
        z6 = tf.nn.relu(z5)
        z7 = tf.nn.avg_pool(
            z6, ksize=[1, 2, 2, 1], padding='VALID', strides=[1, 2, 2, 1])

        z8 = tf.transpose(z7, [0, 3, 1, 2])
        shape = (7 * 7 * 64, 10)
        weight3 = tf.Variable(np.random.normal(
            scale=0.1, size=shape).astype(np.float32))
        #bias = tf.Variable(np.random.normal(scale=0.1, size=shape[-1:]).astype(np.float32))
        z9 = tf.reshape(z8, (-1, shape[0]))
        y = tf.matmul(z9, weight3)  # + bias
        _ = tf.identity(y, name='output')

        sess.run(tf.global_variables_initializer())
        expected = sess.run(y, feed_dict={x: X_val})
        graph_def = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['output'])

    tf.reset_default_graph()
    tf.import_graph_def(graph_def, name='')

    logs.append('saving tf model to onnx! filename is {}'.format(model_name))
    print(logs[-1])

    with tf.Session() as sess:
        onnx_graph = tf2onnx.tfonnx.process_tf_graph(sess.graph,
                                                     input_names=['input:0'],
                                                     output_names=['output:0'],)
        model_proto = onnx_graph.make_model('cnn_model')
        with open(model_name, 'wb') as f:
            f.write(model_proto.SerializeToString())


def onnx2hetu(logs, model_name='tf_cnn_model.onnx'):
    logs.append('loading onnx file to hetu! filename is {}'.format(model_name))
    print(logs[-1])

    x, y = hx.onnx2hetu.load_onnx(model_name)
    logs.append('loading onnx file to hetu PASS!')
    print(logs[-1])
    executor = ht.Executor(
        [y],
        ctx=ctx)
    rand = np.random.RandomState(seed=123)
    datasets = ht.data.mnist()
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # X_val=rand.normal(scale=0.1, size=(20,784)).astype(np.float32)
    #
    X_val = train_set_x[:20, :]

    logs.append(
        'validing models use assert_allclose between hetu and onnxruntime!...')
    print(logs[-1])

    ath = executor.run(
        feed_dict={
            x: X_val})
    sess = rt.InferenceSession(model_name)
    input = sess.get_inputs()[0].name

    pre = sess.run(None, {input: X_val.astype(np.float32)})[0]

    np.testing.assert_allclose(ath[0].asnumpy(), pre, rtol=1e-2)
    logs.append('validing models(cnn) PASS!')
    print(logs[-1])


if __name__ == '__main__':
    logs = []
    tf_model(logs)
    onnx2hetu(logs)

    print('--------print all logs once!-------')
    for i, log in enumerate(logs):
        print(i+1, log)
    print('--------logs end!--------')

    # cnn(executor_ctx=ctx, num_epochs=5, print_loss_val_each_epoch=True)
