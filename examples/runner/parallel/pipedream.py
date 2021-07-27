import hetu as ht

import os
import sys
import time
import argparse
import numpy as np

def fc(x, shape, name, with_relu=True):
    weight = ht.init.random_normal(shape, stddev=0.1, name=name+'_weight')
    bias = ht.init.random_normal(shape[-1:], stddev=0.1, name=name+'_bias')
    x = ht.matmul_op(x, weight)
    x = x + ht.broadcastto_op(bias, x)
    if with_relu:
        x = ht.relu_op(x)
    return x

def make_generator(bs, x, y):
    total_batches = x.shape[0] // bs

    def x_gen_f():
        for i in range(total_batches):
            start = i * bs
            end = (i+1) * bs
            cur_x = x[start:end]
            yield cur_x

    def y_gen_f():
        for i in range(total_batches):
            start = i * bs
            end = (i+1) * bs
            cur_y = y[start:end]
            yield cur_y

    return x_gen_f(), y_gen_f()

if __name__ == "__main__":
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=2000, help='training epochs')
    parser.add_argument('--warmup', type=int, default=2, help='warm up steps excluded from timing')
    parser.add_argument('--batch-size', type=int, default=2048, help='batch size')
    parser.add_argument('--learning-rate', type=float, default=0.0001, help='learning rate')
    args = parser.parse_args()

    datasets = ht.data.mnist()
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # pipeline parallel
    with ht.context(ht.gpu(0)):
        x = ht.Variable(name="dataloader_x", trainable=False)
        activation = fc(x, (784, 1024), 'mlp_fc1', with_relu=True)

    for i in range(1, 7):
        with ht.context(ht.gpu(i)):
            activation = fc(activation, (1024, 1024), 'mlp_fc%d' % (i + 1), with_relu=True)

    with ht.context(ht.gpu(7)):
        y_pred = fc(activation, (1024, 10), 'mlp_fc8', with_relu=True)
        y_ = ht.Variable(name="dataloader_y", trainable=False)
        loss = ht.softmaxcrossentropy_op(y_pred, y_)
        loss = ht.reduce_mean_op(loss, [0])
        opt = ht.optim.SGDOptimizer(learning_rate=args.learning_rate)
        train_op = opt.minimize(loss)
        executor = ht.Executor([loss, train_op], pipedream=True)

    # training
    for epoch in range(args.epochs):
        rand_ind = np.random.randint(train_set_y.shape[0], size=(train_set_y.shape[0],))
        train_set_x = train_set_x[rand_ind]
        train_set_y = train_set_y[rand_ind]
        x_gen, y_gen = make_generator(args.batch_size, train_set_x, train_set_y)

        res = executor.run(feed_dict={x: x_gen, y_: y_gen})
        reduced_res = []
        for elements in res:
            for e in elements:
                if e:
                    reduced_res.append(e[0])
        if reduced_res:
            print("epoch {}, avg loss {}, max loss {}, min loss {}".format(epoch,
                np.mean(reduced_res), np.max(reduced_res), np.min(reduced_res)))
