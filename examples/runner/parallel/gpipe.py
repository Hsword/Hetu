import hetu as ht

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
    parser.add_argument('--epochs', type=int, default=10,
                        help='training epochs')
    parser.add_argument('--warmup', type=int, default=2,
                        help='warm up steps excluded from timing')
    parser.add_argument('--batch-size', type=int,
                        default=1024, help='batch size')
    parser.add_argument('--micro-batches-num', type=int,
                        default=4, help='micro batches number in gpipe')
    parser.add_argument('--learning-rate', type=float,
                        default=0.01, help='learning rate')
    args = parser.parse_args()
    np.random.seed(0)
    datasets = ht.data.mnist()
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # pipeline parallel
    with ht.context(ht.gpu(0)):
        x = ht.dataloader_op([
            ht.Dataloader(train_set_x, args.batch_size, "train"),
            ht.Dataloader(valid_set_x, args.batch_size, "validate"),
        ])
        activation = fc(x, (784, 1024), 'mlp_fc1', with_relu=True)

    for i in range(1, 3):
        with ht.context(ht.gpu(i)):
            activation = fc(activation, (1024, 1024), 'mlp_fc%d' %
                            (i + 1), with_relu=True)

    with ht.context(ht.gpu(3)):
        y_pred = fc(activation, (1024, 10), 'mlp_fc8', with_relu=False)
        y_ = ht.dataloader_op([
            ht.Dataloader(train_set_y, args.batch_size, "train"),
            ht.Dataloader(valid_set_y, args.batch_size, "validate"),
        ])
        loss = ht.softmaxcrossentropy_op(y_pred, y_)
        loss = ht.reduce_mean_op(loss, [0])
        opt = ht.optim.SGDOptimizer(
            learning_rate=args.learning_rate / args.micro_batches_num)
        train_op = opt.minimize(loss)
        executor = ht.Executor({"train": [loss, train_op], "validate": [
                               loss]}, seed=0, pipeline='gpipe')

    def validate():
        val_batch_num = steps = valid_set_x.shape[0] // args.batch_size
        val_loss = []
        for i in range(val_batch_num):
            loss = executor.run("validate", convert_to_numpy_ret_vals=True)
            val_loss.append(loss)
        if executor.rank == 3:
            print("EVAL LOSS: ", np.mean(val_loss))
    # training
    steps = train_set_x.shape[0] // (args.micro_batches_num * args.batch_size)
    for epoch in range(args.epochs):
        loss_vals = []
        if epoch == args.warmup:
            start_time = time.time()
        for step in range(steps):
            ret = executor.run(
                "train", convert_to_numpy_ret_vals=True, batch_num=args.micro_batches_num)
            loss_vals = ret[0]
        if executor.rank == 3:
            print('epoch: {}, mean loss: {}, min loss: {}, max loss: {}'.format(epoch,
                                                                                np.mean(loss_vals), np.min(loss_vals), np.max(loss_vals)))
        validate()

    # if executor.rank == 0:
    #    end_time = time.time()
    #    print("time elapsed for {} epochs: {}s".format(args.epochs-args.warmup, round(end_time-start_time, 3)))
