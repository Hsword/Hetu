import numpy as np
import tensorflow as tf


def tf_rnn(x, y_):
    '''
    RNN model in TensorFlow, for MNIST dataset.

    Parameters:
        x: Variable(tensorflow.python.framework.ops.Tensor), shape (N, dims)
        y_: Variable(tensorflow.python.framework.ops.Tensor), shape (N, num_classes)
    Return:
        loss: Variable(tensorflow.python.framework.ops.Tensor), shape (1,)
        y: Variable(tensorflow.python.framework.ops.Tensor), shape (N, num_classes)
    '''

    print("Building RNN model in tensorflow...")
    diminput = 28
    dimhidden = 128
    dimoutput = 10
    nsteps = 28

    weight1 = tf.Variable(np.random.normal(
        scale=0.1, size=(diminput, dimhidden)).astype(np.float32))
    bias1 = tf.Variable(np.random.normal(
        scale=0.1, size=(dimhidden, )).astype(np.float32))
    weight2 = tf.Variable(np.random.normal(scale=0.1, size=(
        dimhidden + dimhidden, dimhidden)).astype(np.float32))
    bias2 = tf.Variable(np.random.normal(
        scale=0.1, size=(dimhidden, )).astype(np.float32))
    weight3 = tf.Variable(np.random.normal(
        scale=0.1, size=(dimhidden, dimoutput)).astype(np.float32))
    bias3 = tf.Variable(np.random.normal(
        scale=0.1, size=(dimoutput, )).astype(np.float32))
    last_state = tf.zeros((128, dimhidden), dtype=tf.float32)

    for i in range(nsteps):
        cur_x = tf.slice(x, (0, i * diminput), (-1, diminput))
        h = tf.matmul(cur_x, weight1) + bias1

        s = tf.concat([h, last_state], axis=1)
        s = tf.matmul(s, weight2) + bias2
        last_state = tf.nn.relu(s)

    final_state = last_state
    y = tf.matmul(final_state, weight3) + bias3
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)
    loss = tf.reduce_mean(loss)
    return loss, y
