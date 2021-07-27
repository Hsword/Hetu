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
    parser.add_argument('--epochs', type=int, default=100, help='training epochs')
    parser.add_argument('--warmup', type=int, default=2, help='warm up steps excluded from timing')
    parser.add_argument('--batch-size', type=int, default=1024, help='batch size')
    parser.add_argument('--micro-batches-num', type=int, default=8, help='micro batches number in gpipe')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='learning rate')
    args = parser.parse_args()

    datasets = ht.data.mnist()
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # pipeline parallel
    with ht.context(ht.gpu(0)):
        x = ht.Variable(name="dataloader_x", trainable=False)
        activation = fc(x, (784, 1024), 'mlp_fc1', with_relu=True)

    for i in range(1, args.micro_batches_num-1):
        with ht.context(ht.gpu(i)):
            activation = fc(activation, (1024, 1024), 'mlp_fc%d' % (i + 1), with_relu=True)

    with ht.context(ht.gpu(args.micro_batches_num-1)):
        y_pred = fc(activation, (1024, 10), 'mlp_fc8', with_relu=True)
        y_ = ht.Variable(name="dataloader_y", trainable=False)
        loss = ht.softmaxcrossentropy_op(y_pred, y_)
        loss = ht.reduce_mean_op(loss, [0])
        opt = ht.optim.SGDOptimizer(learning_rate=args.learning_rate)
        train_op = opt.minimize(loss)
        executor = ht.Executor([loss, train_op], gpipe=True)

    # training
    steps = train_set_x.shape[0] // (args.micro_batches_num * args.batch_size)
    for epoch in range(args.epochs):
        loss_vals = []
        if epoch == args.warmup:
            start_time = time.time()
        for step in range(steps):
            feed_dicts_list = []
            for i in range(args.micro_batches_num):
                start = (step * args.micro_batches_num + i) * args.batch_size
                end = (step * args.micro_batches_num + i + 1) * args.batch_size
                feed_dicts_list.append({x: train_set_x[start:end], y_: train_set_y[start:end]})
            ret = executor.run(feed_dict=feed_dicts_list, convert_to_numpy_ret_vals=True)
            for r in ret:
                for e in r:
                    if e:
                        loss_vals.append(e[0])
        if executor.rank == args.micro_batches_num-1:
            print('epoch: {}, mean loss: {}, min loss: {}, max loss: {}'.format(epoch,
                np.mean(loss_vals), np.min(loss_vals), np.max(loss_vals)))

    #if executor.rank == 0:
    #    end_time = time.time()
    #    print("time elapsed for {} epochs: {}s".format(args.epochs-args.warmup, round(end_time-start_time, 3)))

