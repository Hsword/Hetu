import hetu as ht

import time
import argparse
import numpy as np


def fc(x, shape, name, with_relu=True, ctx=None):
    weight_save = np.load('std/' + name + '_weight.npy')
    bias_save = np.load('std/' + name + '_bias.npy')
    weight = ht.Variable(value=weight_save, name=name+'_weight', ctx=ctx)
    bias = ht.Variable(value=bias_save, name=name+'_bias', ctx=ctx)
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
    parser.add_argument('--split', type=str, default='left')
    parser.add_argument('--split2', type=str, default=None)
    parser.add_argument('--complex', action='store_true')
    parser.add_argument('--complex2', action='store_true')
    parser.add_argument('--log', default=None)
    args = parser.parse_args()
    assert args.split in ('left', 'right', 'middle',
                          '0', '1', '2', '3', '4', '5')
    assert args.split2 in (None, 'left', 'right',
                           'middle', '0', '1', '2', '3', '4', '5')
    args.complex = not args.split in ('left', 'right', 'middle')
    args.complex2 = not args.split2 in ('left', 'right', 'middle')

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

    context1 = tuple((ht.gpu(0), ht.gpu(1), ht.gpu(2), ht.gpu(
        3))) if args.complex else tuple((ht.gpu(0), ht.gpu(1)))
    with ht.context(context1):
        weight_save = np.load('std/' + 'special_weight.npy')
        weight = ht.Variable(value=weight_save, name='mlp_fc1_weight')
        if args.split == 'left':
            activation = ht.dispatch(activation, (2, 1))
            weight = ht.dispatch(weight, (1, 1))
        elif args.split == 'right':
            activation = ht.dispatch(activation, (1, 1))
            weight = ht.dispatch(weight, (1, 2))
        elif args.split == 'middle':
            activation = ht.dispatch(activation, (1, 2))
            weight = ht.dispatch(weight, (2, 1))
        elif args.split == '0':
            activation = ht.dispatch(activation, (4, 1))
            weight = ht.dispatch(weight, (1, 1))
        elif args.split == '1':
            activation = ht.dispatch(activation, (2, 2))
            weight = ht.dispatch(weight, (2, 1))
        elif args.split == '2':
            activation = ht.dispatch(activation, (2, 1))
            weight = ht.dispatch(weight, (1, 2))
        elif args.split == '3':
            activation = ht.dispatch(activation, (1, 2))
            weight = ht.dispatch(weight, (2, 2))
        elif args.split == '4':
            activation = ht.dispatch(activation, (1, 1))
            weight = ht.dispatch(weight, (1, 4))
        elif args.split == '5':
            activation = ht.dispatch(activation, (1, 4))
            weight = ht.dispatch(weight, (4, 1))
        activation = ht.matmul_op(activation, weight)
        activation = ht.relu_op(activation)

    if args.split2:
        context2 = tuple((ht.gpu(0), ht.gpu(1), ht.gpu(2), ht.gpu(
            3))) if args.complex2 else tuple((ht.gpu(0), ht.gpu(1)))
        with ht.context(context2):
            weight_save = np.load('std/' + 'special_weight2.npy')
            weight = ht.Variable(
                value=weight_save, name='special_mlp_fc2_weight')
            if args.split2 == 'left':
                activation = ht.dispatch(activation, (2, 1))
                weight = ht.dispatch(weight, (1, 1))
            elif args.split2 == 'right':
                activation = ht.dispatch(activation, (1, 1))
                weight = ht.dispatch(weight, (1, 2))
            elif args.split2 == 'middle':
                activation = ht.dispatch(activation, (1, 2))
                weight = ht.dispatch(weight, (2, 1))
            elif args.split2 == '0':
                activation = ht.dispatch(activation, (4, 1))
                weight = ht.dispatch(weight, (1, 1))
            elif args.split2 == '1':
                activation = ht.dispatch(activation, (2, 2))
                weight = ht.dispatch(weight, (2, 1))
            elif args.split2 == '2':
                activation = ht.dispatch(activation, (2, 1))
                weight = ht.dispatch(weight, (1, 2))
            elif args.split2 == '3':
                activation = ht.dispatch(activation, (1, 2))
                weight = ht.dispatch(weight, (2, 2))
            elif args.split2 == '4':
                activation = ht.dispatch(activation, (1, 1))
                weight = ht.dispatch(weight, (1, 4))
            elif args.split == '5':
                activation = ht.dispatch(activation, (1, 4))
                weight = ht.dispatch(weight, (4, 1))
            activation = ht.matmul_op(activation, weight)
            activation = ht.relu_op(activation)

    with ht.context(ht.gpu(1)):
        activation = ht.dispatch(activation, (1, 1))
        # activation = ht.relu_op(activation)
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
        if executor.rank == 1:
            print('step:', step, 'loss:', loss_val)
            results.extend(loss_val)

    end = time.time()
    if executor.rank == 1:
        print("time elapsed for {} steps: {}s".format(
            args.steps-args.warmup, round(end-start, 3)))
        if args.log:
            np.save(args.log, results)
