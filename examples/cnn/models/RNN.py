import hetu as ht
from hetu import init
import numpy as np


def rnn(x, y_):
    '''
    RNN model, for MNIST dataset.

    Parameters:
        x: Variable(hetu.gpu_ops.Node.Node), shape (N, dims)
        y_: Variable(hetu.gpu_ops.Node.Node), shape (N, num_classes)
    Return:
        loss: Variable(hetu.gpu_ops.Node.Node), shape (1,)
        y: Variable(hetu.gpu_ops.Node.Node), shape (N, num_classes)
    '''

    print("Building RNN model...")
    diminput = 28
    dimhidden = 128
    dimoutput = 10
    nsteps = 28

    weight1 = init.random_normal(
        shape=(diminput, dimhidden), stddev=0.1, name='rnn_weight1')
    bias1 = init.random_normal(
        shape=(dimhidden, ), stddev=0.1, name='rnn_bias1')
    weight2 = init.random_normal(
        shape=(dimhidden+dimhidden, dimhidden), stddev=0.1, name='rnn_weight2')
    bias2 = init.random_normal(
        shape=(dimhidden, ), stddev=0.1, name='rnn_bias2')
    weight3 = init.random_normal(
        shape=(dimhidden, dimoutput), stddev=0.1, name='rnn_weight3')
    bias3 = init.random_normal(
        shape=(dimoutput, ), stddev=0.1, name='rnn_bias3')
    last_state = ht.Variable(value=np.zeros((1,)).astype(
        np.float32), name='initial_state', trainable=False)

    for i in range(nsteps):
        cur_x = ht.slice_op(x, (0, i*diminput), (-1, diminput))
        h = ht.matmul_op(cur_x, weight1)
        h = h + ht.broadcastto_op(bias1, h)

        if i == 0:
            last_state = ht.broadcastto_op(last_state, h)
        s = ht.concat_op(h, last_state, axis=1)
        s = ht.matmul_op(s, weight2)
        s = s + ht.broadcastto_op(bias2, s)
        last_state = ht.relu_op(s)

    final_state = last_state
    x = ht.matmul_op(final_state, weight3)
    y = x + ht.broadcastto_op(bias3, x)
    loss = ht.softmaxcrossentropy_op(y, y_)
    loss = ht.reduce_mean_op(loss, [0])
    return loss, y
