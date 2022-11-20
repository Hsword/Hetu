import hetu as ht
from hetu import stream
from hetu import init

import os
import sys
import json
import time
import argparse
import numpy as np
import logging

np.random.seed(123)


def convert_to_one_hot(vals, max_val=0):
    """Helper method to convert label array to one-hot array."""
    if max_val == 0:
        max_val = vals.max() + 1
    one_hot_vals = np.zeros((vals.size, max_val))
    one_hot_vals[np.arange(vals.size), vals] = 1
    return one_hot_vals


def fc(x, shape, name, with_relu=True, ctx=None):
    weight = init.random_normal(
        shape=shape, stddev=0.04, name=name+'_weight', ctx=ctx)
    bias = init.random_normal(
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
    args = parser.parse_args()

    # init and opt for both ranks
    comm = ht.wrapped_mpi_nccl_init()
    device_id = comm.dev_id
    print("mpi_nccl init for gpu device: {}".format(device_id))
    executor_ctx = ht.gpu(device_id)
    opt = ht.optim.SGDOptimizer(learning_rate=args.learning_rate)

    # init logger
    logger = logging.getLogger()
    ch = logging.StreamHandler()
    formatter = logging.Formatter('[rank{}, PID{}]'.format(
        device_id, os.getpid()) + ' %(asctime)s: %(message)s')
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    log = logger.warning

    # nccl communicate stream for pipeline_send/receive
    communicate_stream = stream.create_stream_handle(executor_ctx)

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

    x = ht.Variable(name="dataloader_x", trainable=False)
    y_ = ht.Variable(name="dataloader_y", trainable=False)

    # model parallel
    if comm.myRank.value == 0:
        # rank0

        # forward
        activation = fc(x, (784, 1024), 'mlp_fc1', with_relu=True,
                        ctx=ht.gpu(comm.localRank.value))
        activation = fc(activation, (1024, 2048), 'mlp_fc2',
                        with_relu=True, ctx=ht.gpu(comm.localRank.value))
        activation = fc(activation, (2048, 1024), 'mlp_fc3',
                        with_relu=True, ctx=ht.gpu(comm.localRank.value))
        activation_send_op = ht.pipeline_send_op(
            activation, 1, comm)

        # backward
        gradient_receive_op = ht.pipeline_receive_op(
            1, comm, ctx=executor_ctx)
        required_vars = opt.get_var_list(activation)
        opt.params = required_vars
        grads = ht.gradients(activation, required_vars,
                             insert_grad=gradient_receive_op)
        train_op = ht.optim.OptimizerOp(grads, opt)

        executor = ht.Executor(
            [activation_send_op, train_op], ctx=executor_ctx)

    elif comm.myRank.value != 7:
        # from rank1 to rank6
        previous_rank = comm.myRank.value - 1
        next_rank = comm.myRank.value + 1

        # 1. receive activation from previous rank
        activation_receive_op = ht.pipeline_receive_op(
            previous_rank, comm, ctx=executor_ctx)
        # forward
        activation = fc(activation_receive_op, (1024, 2048), 'mlp_fc1',
                        with_relu=True, ctx=ht.gpu(comm.localRank.value))
        activation = fc(activation, (2048, 2048), 'mlp_fc2',
                        with_relu=True, ctx=ht.gpu(comm.localRank.value))
        activation = fc(activation, (2048, 1024), 'mlp_fc3',
                        with_relu=True, ctx=ht.gpu(comm.localRank.value))

        # 2. send activation to next rank
        activation_send_op = ht.pipeline_send_op(
            activation, next_rank, comm, ctx=executor_ctx)

        # 3. receive gradients from next rank
        gradient_receive_op = ht.pipeline_receive_op(
            next_rank, comm, ctx=executor_ctx)
        # backward
        required_vars = opt.get_var_list(activation)
        opt.params = required_vars
        required_vars = [activation_receive_op] + required_vars
        grads = ht.gradients(activation, required_vars,
                             insert_grad=gradient_receive_op)
        train_op = ht.optim.OptimizerOp(grads[1:], opt)

        # 4. send gradients to previous rank
        sendback_grad_op = ht.pipeline_send_op(
            grads[0], previous_rank, comm)

        executor = ht.Executor(
            [activation_send_op, sendback_grad_op, train_op], ctx=executor_ctx)

    else:
        # rank7
        activation_receive_op = ht.pipeline_receive_op(
            6, comm, ctx=executor_ctx)

        # forward
        activation = fc(activation_receive_op, (1024, 2048), 'mlp_fc1',
                        with_relu=True, ctx=ht.gpu(comm.localRank.value))
        activation = fc(activation, (2048, 1024), 'mlp_fc2',
                        with_relu=True, ctx=ht.gpu(comm.localRank.value))
        y_pred = fc(activation, (1024, 10), 'mlp_fc3', with_relu=False)
        loss = ht.softmaxcrossentropy_op(y_pred, y_)
        loss = ht.reduce_mean_op(loss, [0])

        # backward
        required_vars = opt.get_var_list(loss)
        opt.params = required_vars
        required_vars = [activation_receive_op] + required_vars
        grads = ht.gradients(loss, required_vars)
        train_op = ht.optim.OptimizerOp(grads[1:], opt)

        sendback_grad_op = ht.pipeline_send_op(
            grads[0], 6, comm)
        executor = ht.Executor(
            [loss, sendback_grad_op, train_op], ctx=executor_ctx)

    # training
    for step in range(args.steps):
        if step == args.warmup:
            start = time.time()
        if comm.myRank.value == 0:
            log("step {}:".format(step))
        if comm.myRank.value == 0:
            executor.run(feed_dict={x: value_x_list[step % batch_num]})
            log("gpu0 ok")
        elif comm.myRank.value == 7:
            loss, _, _ = executor.run(
                feed_dict={y_: value_y_list[step % batch_num]}, convert_to_numpy_ret_vals=True)
            log("gpu7 ok, loss: {}".format(loss[0]))
        else:
            executor.run()
            log("gpu{} ok".format(comm.myRank.value))

    # comm.stream.sync()
    if communicate_stream:
        communicate_stream.sync()

    end = time.time()
    log("time elapsed for {} steps: {}s".format(
        args.steps-args.warmup, round(end-start, 3)))
