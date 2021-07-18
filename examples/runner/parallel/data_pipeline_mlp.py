import hetu as ht

import os
import time
import argparse
import numpy as np


def fc(x, shape, name, with_relu=True):
    weight = ht.init.random_normal(shape, stddev=0.04, name=name+'_weight')
    bias = ht.init.random_normal(shape[-1:], stddev=0.04, name=name+'_bias')
    x = ht.matmul_op(x, weight)
    x = x + ht.broadcastto_op(bias, x)
    if with_relu:
        x = ht.relu_op(x)
    return x


if __name__ == "__main__":
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--warmup', type=int, default=1,
                        help='warm up steps excluded from timing')
    parser.add_argument('--batch-size', type=int,
                        default=10000, help='batch size')
    parser.add_argument('--learning-rate', type=float,
                        default=0.01, help='learning rate')
    args = parser.parse_args()

    datasets = ht.data.mnist()
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    with ht.context("gpu:0,gpu:4"):
        x = ht.Variable(name="dataloader_x", trainable=False)
        activation = fc(x, (784, 1024), 'mlp_fc0', with_relu=True)

    with ht.context("gpu:1,gpu:5"):
        activation = fc(activation, (1024, 1024), 'mlp_fc1', with_relu=True)
        activation = fc(activation, (1024, 1024), 'mlp_fc11', with_relu=True)

    with ht.context("gpu:2,gpu:6"):
        activation = fc(activation, (1024, 1024), 'mlp_fc2', with_relu=True)
        activation = fc(activation, (1024, 1024), 'mlp_fc22', with_relu=True)

    with ht.context("gpu:3,gpu:7"):
        y_pred = fc(activation, (1024, 10), 'mlp_fc3', with_relu=True)
        y_ = ht.Variable(name="dataloader_y", trainable=False)
        loss = ht.softmaxcrossentropy_op(y_pred, y_)
        loss = ht.reduce_mean_op(loss, [0])
        opt = ht.optim.SGDOptimizer(learning_rate=args.learning_rate)
        train_op = opt.minimize(loss)
        executor = ht.Executor([loss, train_op])

    print_devices = [3, 7]

    # training
    steps = train_set_x.shape[0] // args.batch_size
    for step in range(steps):
        start = step * args.batch_size
        end = start + args.batch_size
        loss_val, _ = executor.run(feed_dict={
                                   x: train_set_x[start:end], y_: train_set_y[start:end]}, convert_to_numpy_ret_vals=True)
        if executor.local_rank in print_devices:
            print('[step {}]: loss: {}'.format(step, loss_val[0]))
