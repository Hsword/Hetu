import hetu as ht

import time
import argparse
import os
import numpy as np


def fc(x, shape, name, with_relu=True):
    weight_save = np.random.normal(0, 0.04, size=shape)
    bias_save = np.random.normal(0, 0.04, size=shape[-1:])
    weight = ht.Variable(value=weight_save, name=name+'_weight')
    bias = ht.Variable(value=bias_save, name=name+'_bias')
    global args
    if args.save:
        np.save('std/' + name + '_weight.npy', weight_save)
        np.save('std/' + name + '_bias.npy', bias_save)
    x = ht.matmul_op(x, weight)
    x = x + ht.broadcastto_op(bias, x)
    if with_relu:
        x = ht.relu_op(x)
    return x


if __name__ == "__main__":
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=8, help='training steps')
    parser.add_argument('--warmup', type=int, default=2,
                        help='warm up steps excluded from timing')
    parser.add_argument('--batch-size', type=int, default=8, help='batch size')
    parser.add_argument('--learning-rate', type=float,
                        default=0.01, help='learning rate')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--more', action='store_true')
    parser.add_argument('--log', default=None)
    global args
    args = parser.parse_args()
    if args.save:
        if not os.path.exists('std'):
            os.mkdir('std')

    # dataset
    datasets = ht.data.mnist()
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    batch_size = 10000
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
        activation = fc(x, (784, 1024), 'mlp_fc1', with_relu=True)

        weight_save = np.random.normal(0, 0.04, size=(1024, 2048))
        if args.save:
            np.save('std/' + 'special_weight.npy', weight_save)
        weight = ht.Variable(value=weight_save, name='special_mlp_fc1_weight')
        activation = ht.matmul_op(activation, weight)
        activation = ht.relu_op(activation)
        if args.more:
            weight_save = np.random.normal(0, 0.04, size=(2048, 2048))
            if args.save:
                np.save('std/' + 'special_weight2.npy', weight_save)
            weight = ht.Variable(
                value=weight_save, name='special_mlp_fc2_weight')
            activation = ht.matmul_op(activation, weight)
            activation = ht.relu_op(activation)

        y_pred = fc(activation, (2048, 10), 'mlp_fc2', with_relu=False)
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
        print('step:', step, 'loss:', loss_val)
        results.extend(loss_val)

    end = time.time()
    print("time elapsed for {} steps: {}s".format(
        args.steps-args.warmup, round(end-start, 3)))
    if args.log:
        np.save(args.log, results)
