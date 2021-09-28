import hetu as ht

import os
import sys
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
    parser.add_argument('--epochs', type=int, default=5, help='training epochs')
    parser.add_argument('--log_every', type=int, default=240, help='warm up steps excluded from timing')
    parser.add_argument('--batch-size', type=int, default=1024, help='batch size')
    parser.add_argument('--learning-rate', type=float, default=0.01, help='learning rate')
    args = parser.parse_args()
    np.random.seed(0)

    datasets = ht.data.mnist()
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    iterations = len(train_set_x) // args.batch_size * args.epochs

    # pipeline parallel
    with ht.context([ht.cpu(0), ht.gpu(1)]):
        x = ht.dataloader_op([
            ht.Dataloader(train_set_x, args.batch_size, "train"),
            ht.Dataloader(valid_set_x, args.batch_size, "validate"),
        ])
        activation = fc(x, (784, 1024), 'mlp_fc1', with_relu=True)

    for i in range(2, 3):
        with ht.context([ht.cpu(0), ht.gpu(i)]):
            activation = fc(activation, (1024, 1024), 'mlp_fc%d' % (i + 1), with_relu=True)

    with ht.context([ht.cpu(0), ht.gpu(3)]):
        y_pred = fc(activation, (1024, 10), 'mlp_fc8', with_relu=False)
        y_ = ht.dataloader_op([
            ht.Dataloader(train_set_y, args.batch_size, "train"),
            ht.Dataloader(valid_set_y, args.batch_size, "validate"),
        ])
        loss = ht.softmaxcrossentropy_op(y_pred, y_)
        loss = ht.reduce_mean_op(loss, [0])
        opt = ht.optim.SGDOptimizer(learning_rate=args.learning_rate)
        train_op = opt.minimize(loss)
        executor = ht.Executor({"train" : [loss, train_op], "validate" : [loss]}, seed=0, pipeline="pipedream")

    def validate():
        val_batch_num = steps = valid_set_x.shape[0] // args.batch_size
        val_loss = []
        for i in range(val_batch_num):
            loss = executor.run("validate", convert_to_numpy_ret_vals=True)
            val_loss.append(loss)
        if executor.rank == 2:
            print("EVAL LOSS: ", np.mean(val_loss))

    # training
    for _ in range(iterations // args.log_every):
        start = time.time()
        res = executor.run("train", batch_num = args.log_every)
        time_used = time.time() - start
        reduced_res = []
        for elements in res:
            for e in elements:
                if e:
                    reduced_res.append(e[0])
        if reduced_res:
            print("TRAIN avg loss {}, max loss {}, min loss {} time {}".format(
                    np.mean(reduced_res), np.max(reduced_res), np.min(reduced_res), time_used))

        validate()
    ht.worker_finish()
