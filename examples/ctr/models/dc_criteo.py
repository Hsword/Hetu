import hetu as ht
from hetu import init

import numpy as np
import time


def residual_layer(x0, input_dim, hidden_dim):

    embedding_len = input_dim
    weight_1 = init.random_normal(
        shape=(input_dim, hidden_dim), stddev=0.1, name='weight_1')
    bias_1 = init.random_normal(shape=(hidden_dim,), stddev=0.1, name='bias_1')
    weight_2 = init.random_normal(
        shape=(hidden_dim, input_dim), stddev=0.1, name='weight_2')
    bias_2 = init.random_normal(shape=(input_dim,), stddev=0.1, name='bias_2')

    x0w = ht.matmul_op(x0, weight_1)  # (batch, hidden_dim)
    x0w_b = x0w + ht.broadcastto_op(bias_1, x0w)

    relu1 = ht.relu_op(x0w_b)
    x1w = ht.matmul_op(relu1, weight_2)  # (batch, input_dim)
    x1w_b = x1w + ht.broadcastto_op(bias_2, x1w)
    residual = x1w_b + x0
    y = ht.relu_op(residual)
    return y


def build_residual_layers(x0, input_dim, hidden_dim, num_layers=3):
    for i in range(num_layers):
        x0 = residual_layer(x0, input_dim, hidden_dim)
    return x0


def dc_criteo(dense_input, sparse_input, y_):

    feature_dimension = 33762577
    embedding_size = 8
    learning_rate = 0.001

    Embedding = init.random_normal(
        [feature_dimension, embedding_size], stddev=0.01, name="snd_order_embedding")
    sparse_input = ht.embedding_lookup_op(Embedding, sparse_input)
    sparse_input = ht.array_reshape_op(sparse_input, (-1, 26*embedding_size))

    # dc_model
    x = ht.concat_op(sparse_input, dense_input, axis=1)

    input_dim = 26 * 8 + 13
    hidden_dim = input_dim
    residual_out = build_residual_layers(
        x, input_dim, hidden_dim, num_layers=5)

    W4 = init.random_normal([26*embedding_size + 13, 1], stddev=0.1, name="W4")
    y = ht.matmul_op(residual_out, W4)
    y = ht.sigmoid_op(y)

    loss = ht.binarycrossentropy_op(y, y_)
    loss = ht.reduce_mean_op(loss, [0])
    opt = ht.optim.SGDOptimizer(learning_rate=learning_rate)
    train_op = opt.minimize(loss)

    return loss, y, y_, train_op
