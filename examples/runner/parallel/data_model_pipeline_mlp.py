import hetu as ht

import time
import argparse


def fc(x, shape, name, with_relu=True, ctx=None):
    weight = ht.init.random_normal(
        shape=shape, stddev=0.04, name=name+'_weight', ctx=ctx)
    bias = ht.init.random_normal(
        shape=shape[-1:], stddev=0.04, name=name+'_bias', ctx=ctx)
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
                        default=0.00001, help='learning rate')
    parser.add_argument('--split', type=str, default='left',
                        help='left, middle, right')
    args = parser.parse_args()
    assert args.split in ('left', 'middle', 'right')

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
    with ht.context([ht.gpu(0), ht.gpu(4)]):
        x = ht.Variable(name="dataloader_x", trainable=False)
        activation = fc(x, (784, 1024), 'mlp_fc1', with_relu=True)
        activation = fc(activation, (1024, 2048), 'mlp_fc2', with_relu=True)
        activation = fc(activation, (2048, 1024), 'mlp_fc3', with_relu=True)
        if args.split == 'left':
            activation = ht.dispatch(activation, (2, 1))
            weight = ht.dispatch(ht.init.random_normal(
                shape=(1024, 2048), stddev=0.04, name='mlp_fc1_weight'), (1, 1))
        elif args.split == 'right':
            activation = ht.dispatch(activation, (1, 1))
            weight = ht.dispatch(ht.init.random_normal(
                shape=(1024, 2048), stddev=0.04, name='mlp_fc1_weight'), (1, 2))
        else:
            activation = ht.dispatch(activation, (1, 2))
            weight = ht.dispatch(ht.init.random_normal(
                shape=(1024, 2048), stddev=0.04, name='mlp_fc1_weight'), (2, 1))

    with ht.context([(ht.gpu(1), ht.gpu(2)), (ht.gpu(5), ht.gpu(6))]):
        activation = ht.matmul_op(activation, weight)
        activation = ht.dispatch(activation, (1, 1))

    with ht.context([ht.gpu(3), ht.gpu(7)]):
        activation = ht.relu_op(activation)
        activation = fc(activation, (2048, 2048), 'mlp_fc2', with_relu=True)
        activation = fc(activation, (2048, 1024), 'mlp_fc3', with_relu=True)
        y_pred = fc(activation, (1024, 10), 'mlp_fc3', with_relu=False)
        y_ = ht.Variable(name="dataloader_y", trainable=False)
        loss = ht.softmaxcrossentropy_op(y_pred, y_)
        loss = ht.reduce_mean_op(loss, [0])
        opt = ht.optim.SGDOptimizer(learning_rate=args.learning_rate)
        train_op = opt.minimize(loss)

        executor = ht.Executor([loss, train_op])

    # training
    for step in range(args.steps):
        if step == args.warmup:
            start = time.time()
        loss_val, _ = executor.run(feed_dict={
                                   x: value_x_list[step % batch_num], y_: value_y_list[step % batch_num]}, convert_to_numpy_ret_vals=True)
        if executor.rank == 3:
            print('step:', step, 'loss:', loss_val)

    end = time.time()
    if executor.rank == 3:
        print("time elapsed for {} steps: {}s".format(
            args.steps-args.warmup, round(end-start, 3)))
