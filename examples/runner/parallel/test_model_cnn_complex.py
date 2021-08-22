import hetu as ht

import time
import argparse
import os
import numpy as np


def conv_relu(x, shape, name):
    weight_save = np.load('std/' + name + '_cnn_weight.npy')
    weight = ht.Variable(value=weight_save, name=name+'_cnn_weight')
    x = ht.conv2d_op(x, weight, padding=2, stride=1)
    x = ht.relu_op(x)
    return x


def fc(x, shape, name, with_relu=True):
    weight_save = np.load('std/' + name + '_weight.npy')
    bias_save = np.load('std/' + name + '_bias.npy')
    weight = ht.Variable(value=weight_save, name=name+'_weight')
    bias = ht.Variable(value=bias_save, name=name+'_bias')
    x = ht.matmul_op(x, weight)
    x = x + ht.broadcastto_op(bias, x)
    if with_relu:
        x = ht.relu_op(x)
    return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=8, help='training steps')
    parser.add_argument('--warmup', type=int, default=2,
                        help='warm up steps excluded from timing')
    parser.add_argument('--batch-size', type=int, default=8, help='batch size')
    parser.add_argument('--learning-rate', type=float,
                        default=0.01, help='learning rate')
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--split2', type=str, default=None)
    parser.add_argument('--revert', action='store_true')
    parser.add_argument('--log', default=None)
    args = parser.parse_args()
    assert args.split in tuple(range(5))
    assert args.split2 in (None, 'left', 'right', 'middle')

    # dataset
    datasets = ht.data.mnist()
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    batch_size = 5000
    batch_num = 5
    value_x_list = []
    value_y_list = []
    for i in range(batch_num):
        start = i * batch_size
        ending = (i+1) * batch_size
        value_x_list.append(train_set_x[start:ending])
        value_y_list.append(train_set_y[start:ending])

    # model parallel
    with ht.context(ht.gpu(0)):
        x = ht.Variable(name="dataloader_x", trainable=False)
        activation = ht.array_reshape_op(x, [-1, 1, 28, 28])
        activation = conv_relu(activation, (32, 1, 5, 5), 'cnn')

    if args.revert and args.split2:
        with ht.context((ht.gpu(6), ht.gpu(7))):
            weight_save = np.load('std/' + 'special_cnn_weight.npy')
            weight = ht.Variable(
                value=weight_save, name='special_cnn_weight')
            if args.split == 'left':
                activation = ht.dispatch(activation, (2, 1))
                weight = ht.dispatch(weight, (1, 1))
            elif args.split == 'right':
                activation = ht.dispatch(activation, (1, 1))
                weight = ht.dispatch(weight, (2, 1))
            else:
                activation = ht.dispatch(activation, (1, 2))
                weight = ht.dispatch(weight, (1, 2))
            activation = ht.conv2d_op(activation, weight, padding=2, stride=1)

    with ht.context((ht.gpu(1), ht.gpu(2), ht.gpu(4), ht.gpu(5))):
        if args.revert:
            weight_save = np.load('std/' + 'special_cnn_weight2.npy')
            weight = ht.Variable(value=weight_save, name='special_cnn_weight2')
        else:
            weight_save = np.load('std/' + 'special_cnn_weight.npy')
            weight = ht.Variable(value=weight_save, name='special_cnn_weight')
        if args.split == 0:
            activation = ht.dispatch(activation, (4, 1))
            weight = ht.dispatch(weight, (1, 1))
        elif args.split == 1:
            activation = ht.dispatch(activation, (2, 2))
            weight = ht.dispatch(weight, (1, 2))
        elif args.split == 2:
            activation = ht.dispatch(activation, (2, 1))
            weight = ht.dispatch(weight, (2, 1))
        elif args.split == 3:
            activation = ht.dispatch(activation, (1, 2))
            weight = ht.dispatch(weight, (2, 2))
        elif args.split == 4:
            activation = ht.dispatch(activation, (1, 1))
            weight = ht.dispatch(weight, (4, 1))
        activation = ht.conv2d_op(activation, weight, padding=2, stride=1)

    if not args.revert and args.split2:
        with ht.context((ht.gpu(6), ht.gpu(7))):
            weight_save = np.load('std/' + 'special_cnn_weight2.npy')
            weight = ht.Variable(
                value=weight_save, name='special_cnn_weight2')
            if args.split == 'left':
                activation = ht.dispatch(activation, (2, 1))
                weight = ht.dispatch(weight, (1, 1))
            elif args.split == 'right':
                activation = ht.dispatch(activation, (1, 1))
                weight = ht.dispatch(weight, (2, 1))
            else:
                activation = ht.dispatch(activation, (1, 2))
                weight = ht.dispatch(weight, (1, 2))
            activation = ht.conv2d_op(activation, weight, padding=2, stride=1)

    with ht.context(ht.gpu(3)):
        activation = ht.dispatch(activation, (1, 1))
        activation = ht.relu_op(activation)
        activation = ht.array_reshape_op(activation, (-1, 28 * 28 * 64))
        y_pred = fc(activation, (28 * 28 * 64, 10),
                    'mlp_fc', with_relu=False)
        y_ = ht.Variable(name="dataloader_y", trainable=False)
        loss = ht.softmaxcrossentropy_op(y_pred, y_)
        loss = ht.reduce_mean_op(loss, [0])
        opt = ht.optim.SGDOptimizer(learning_rate=args.learning_rate)
        train_op = opt.minimize(loss)

        executor = ht.Executor([loss, train_op])

    # training
    results = []
    for step in range(args.steps):
        if step == args.warmup:
            start = time.time()
        loss_val, _ = executor.run(feed_dict={
                                   x: value_x_list[step % batch_num], y_: value_y_list[step % batch_num]}, convert_to_numpy_ret_vals=True)
        if executor.rank == 3:
            print('step:', step, 'loss:', loss_val)
            results.extend(loss_val)

    end = time.time()
    if executor.rank == 3:
        print("time elapsed for {} steps: {}s".format(
            args.steps-args.warmup, round(end-start, 3)))
        if args.log:
            np.save(args.log, results)
