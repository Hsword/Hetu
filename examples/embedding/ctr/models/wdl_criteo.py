import hetu as ht
from hetu import init

import numpy as np
import time


def wdl_criteo(dense_input, sparse_input, y_):
    feature_dimension = 33762577
    embedding_size = 128
    learning_rate = 0.01
    Embedding = init.random_normal(
        [feature_dimension, embedding_size], stddev=0.01, name="snd_order_embedding", ctx=ht.cpu(0))
    sparse_input = ht.embedding_lookup_op(
        Embedding, sparse_input, ctx=ht.cpu(0))
    sparse_input = ht.array_reshape_op(sparse_input, (-1, 26*embedding_size))

    # DNN
    flatten = dense_input
    W1 = init.random_normal([13, 256], stddev=0.01, name="W1")
    W2 = init.random_normal([256, 256], stddev=0.01, name="W2")
    W3 = init.random_normal([256, 256], stddev=0.01, name="W3")

    W4 = init.random_normal(
        [256 + 26*embedding_size, 1], stddev=0.01, name="W4")

    fc1 = ht.matmul_op(flatten, W1)
    relu1 = ht.relu_op(fc1)
    fc2 = ht.matmul_op(relu1, W2)
    relu2 = ht.relu_op(fc2)
    y3 = ht.matmul_op(relu2, W3)

    y4 = ht.concat_op(sparse_input, y3, axis=1)
    y = ht.matmul_op(y4, W4)
    y = ht.sigmoid_op(y)

    loss = ht.binarycrossentropy_op(y, y_)
    loss = ht.reduce_mean_op(loss, [0])
    opt = ht.optim.SGDOptimizer(learning_rate=learning_rate)
    train_op = opt.minimize(loss)

    return loss, y, y_, train_op
