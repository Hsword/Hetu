import hetu as ht
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


# ctx=ht.gpu(0)
ctx = ht.cpu(0)
batch_size = 20


def tf_model(logs, model_name='rnn',):
    diminput = 28
    dimhidden = 128
    dimoutput = 10
    nsteps = 28

    logs.append(
        'Building rnn(nsteps={}) models based on tensorflow....'.format(nsteps))
    print(logs[-1])

    with tf.Session() as sess:
        x = tf.placeholder(dtype=tf.float32, shape=(
            batch_size, 784,), name='input')

        weight1 = tf.Variable(np.random.normal(
            scale=0.1, size=(diminput, dimhidden)).astype(np.float32))
        bias1 = tf.Variable(np.random.normal(
            scale=0.1, size=(dimhidden,)).astype(np.float32))
        weight2 = tf.Variable(np.random.normal(scale=0.1, size=(
            dimhidden + dimhidden, dimhidden)).astype(np.float32))
        bias2 = tf.Variable(np.random.normal(
            scale=0.1, size=(dimhidden,)).astype(np.float32))
        weight3 = tf.Variable(np.random.normal(
            scale=0.1, size=(dimhidden, dimoutput)).astype(np.float32))
        bias3 = tf.Variable(np.random.normal(
            scale=0.1, size=(dimoutput,)).astype(np.float32))
        # last_state = tf.zeros((tf.shape(x)[0], dimhidden), dtype=tf.float32)
        last_state = tf.zeros((batch_size, dimhidden), dtype=tf.float32)

        for i in range(nsteps):
            cur_x = tf.slice(x, (0, i * diminput), (-1, diminput))
            h = tf.matmul(cur_x, weight1) + bias1

            s = tf.concat([h, last_state], axis=1)
            s = tf.matmul(s, weight2) + bias2
            last_state = tf.nn.relu(s)

        final_state = last_state
        y = tf.matmul(final_state, weight3) + bias3

        _ = tf.identity(y, name='output')

        sess.run(tf.global_variables_initializer())
        # expected = sess.run(y, feed_dict={x: X_val})
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
        model_proto = onnx_graph.make_model('dnn_model')
        with open(model_name, 'wb') as f:
            f.write(model_proto.SerializeToString())


def onnx2hetu(logs, model_name=''):
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
    logs.append('validing models(rnn) PASS!')
    print(logs[-1])


if __name__ == '__main__':
    logs = []
    tf_model(logs, model_name='tf_rnn_model.onnx')
    onnx2hetu(logs, model_name='tf_rnn_model.onnx')

    print('--------print all logs once!-------')
    for i, log in enumerate(logs):
        print(i+1, log)
    print('--------logs end!--------')

    # cnn(executor_ctx=ctx, num_epochs=5, print_loss_val_each_epoch=True)
