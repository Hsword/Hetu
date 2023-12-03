import hetu as ht
from models import MLP

import os
import numpy as np
import argparse
import json
from time import time


if __name__ == "__main__":
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='local',
                        help='[local, lps(localps), lar(localallreduce), rps(remoteps), rar]')
    parser.add_argument('--batch-size', type=int,
                        default=128, help='batch size')
    parser.add_argument('--learning-rate', type=float,
                        default=0.1, help='learning rate')
    parser.add_argument('--opt', type=str, default='sgd',
                        help='optimizer to be used, default sgd; sgd / momentum / adagrad / adam')
    parser.add_argument('--num-epochs', type=int,
                        default=10, help='epoch number')
    parser.add_argument('--validate', action='store_true',
                        help='whether to use validation')
    parser.add_argument('--timing', action='store_true',
                        help='whether to time the training phase')
    args = parser.parse_args()

    dataset = 'MNIST'

    assert args.opt in ['sgd', 'momentum', 'nesterov',
                        'adagrad', 'adam'], 'Optimizer not supported!'
    if args.opt == 'sgd':
        print('Use SGD Optimizer.')
        opt = ht.optim.SGDOptimizer(learning_rate=args.learning_rate)
    elif args.opt == 'momentum':
        print('Use Momentum Optimizer.')
        opt = ht.optim.MomentumOptimizer(learning_rate=args.learning_rate)
    elif args.opt == 'nesterov':
        print('Use Nesterov Momentum Optimizer.')
        opt = ht.optim.MomentumOptimizer(
            learning_rate=args.learning_rate, nesterov=True)
    elif args.opt == 'adagrad':
        print('Use AdaGrad Optimizer.')
        opt = ht.optim.AdaGradOptimizer(
            learning_rate=args.learning_rate, initial_accumulator_value=0.1)
    else:
        print('Use Adam Optimizer.')
        opt = ht.optim.AdamOptimizer(learning_rate=args.learning_rate)

    # data loading
    print('Loading %s data...' % dataset)
    if dataset == 'MNIST':
        datasets = ht.data.mnist()
        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]
        # train_set_x: (50000, 784), train_set_y: (50000,)
        # valid_set_x: (10000, 784), valid_set_y: (10000,)
        # x_shape = (args.batch_size, 784)
        # y_shape = (args.batch_size, 10)

    # model definition
    ctx = {
        'local': ht.gpu(0),
        'lps': [ht.cpu(0), ht.gpu(0), ht.gpu(1), ht.gpu(4), ht.gpu(5)],
        'lar': [ht.gpu(1), ht.gpu(2), ht.gpu(3), ht.gpu(6)],
        'rps': ['cpu:0', 'node1:gpu:0', 'node1:gpu:2', 'node1:gpu:4', 'node1:gpu:6', 'node2:gpu:1', 'node2:gpu:3'],
        'rar': ['node1:gpu:0', 'node1:gpu:2', 'node1:gpu:4', 'node1:gpu:6', 'node2:gpu:1', 'node2:gpu:3']
    }[args.config]
    with ht.context(ctx):
        print('Building model...')
        x = ht.dataloader_op([
            ht.Dataloader(train_set_x, args.batch_size, 'train'),
            ht.Dataloader(valid_set_x, args.batch_size, 'validate'),
        ])
        y_ = ht.dataloader_op([
            ht.Dataloader(train_set_y, args.batch_size, 'train'),
            ht.Dataloader(valid_set_y, args.batch_size, 'validate'),
        ])

        loss, y = MLP.mlp(x, y_)
        train_op = opt.minimize(loss)

        executor = ht.Executor(
            {'train': [loss, y, train_op], 'validate': [loss, y, y_]})
        n_train_batches = executor.get_batch_num('train')
        n_valid_batches = executor.get_batch_num('validate')

    # training
    print("Start training loop...")
    for i in range(args.num_epochs):
        print("Epoch %d" % i)
        loss_all = 0
        if args.timing:
            start = time()
        for minibatch_index in range(n_train_batches):
            loss_val, predict_y, _ = executor.run('train')
            loss_val = loss_val.asnumpy()
            loss_all += loss_val * x.dataloaders['train'].last_batch_size
        loss_all /= len(train_set_x)
        print("Loss = %f" % loss_all)
        if args.timing:
            end = time()
            print("Time = %f" % (end - start))

        if args.validate:
            correct_predictions = []
            for minibatch_index in range(n_valid_batches):
                loss_val, valid_y_predicted, y_val = executor.run(
                    'validate', convert_to_numpy_ret_vals=True)
                correct_prediction = np.equal(
                    np.argmax(y_val, 1),
                    np.argmax(valid_y_predicted, 1)).astype(np.float32)
                correct_predictions.extend(correct_prediction)
            accuracy = np.mean(correct_predictions)
            print("Validation accuracy = %f" % accuracy)
