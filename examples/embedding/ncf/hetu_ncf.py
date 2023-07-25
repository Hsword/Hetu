import hetu as ht
from hetu import init

import numpy as np


def neural_mf(user_input, item_input, y_, num_users, num_items):
    embed_dim = 8
    layers = [64, 32, 16, 8]
    learning_rate = 0.01

    User_Embedding = init.random_normal(
        (num_users, embed_dim + layers[0] // 2), stddev=0.01, name="user_embed", ctx=ht.cpu(0))
    Item_Embedding = init.random_normal(
        (num_items, embed_dim + layers[0] // 2), stddev=0.01, name="item_embed", ctx=ht.cpu(0))

    user_latent = ht.embedding_lookup_op(
        User_Embedding, user_input, ctx=ht.cpu(0))
    item_latent = ht.embedding_lookup_op(
        Item_Embedding, item_input, ctx=ht.cpu(0))

    mf_user_latent = ht.slice_op(user_latent, (0, 0), (-1, embed_dim))
    mlp_user_latent = ht.slice_op(user_latent, (0, embed_dim), (-1, -1))
    mf_item_latent = ht.slice_op(item_latent, (0, 0), (-1, embed_dim))
    mlp_item_latent = ht.slice_op(item_latent, (0, embed_dim), (-1, -1))

    W1 = init.random_normal((layers[0], layers[1]), stddev=0.1, name='W1')
    W2 = init.random_normal((layers[1], layers[2]), stddev=0.1, name='W2')
    W3 = init.random_normal((layers[2], layers[3]), stddev=0.1, name='W3')
    W4 = init.random_normal((embed_dim + layers[3], 1), stddev=0.1, name='W4')

    mf_vector = ht.mul_op(mf_user_latent, mf_item_latent)
    mlp_vector = ht.concat_op(mlp_user_latent, mlp_item_latent, axis=1)
    fc1 = ht.matmul_op(mlp_vector, W1)
    relu1 = ht.relu_op(fc1)
    fc2 = ht.matmul_op(relu1, W2)
    relu2 = ht.relu_op(fc2)
    fc3 = ht.matmul_op(relu2, W3)
    relu3 = ht.relu_op(fc3)
    concat_vector = ht.concat_op(mf_vector, relu3, axis=1)
    y = ht.matmul_op(concat_vector, W4)
    y = ht.sigmoid_op(y)
    loss = ht.binarycrossentropy_op(y, y_)
    loss = ht.reduce_mean_op(loss, [0])
    opt = ht.optim.SGDOptimizer(learning_rate=learning_rate)
    train_op = opt.minimize(loss)
    return loss, y, train_op
