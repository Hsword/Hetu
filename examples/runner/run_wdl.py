import hetu as ht
from hetu.launcher import launch

import os
import numpy as np
import yaml
import time
import argparse
from tqdm import tqdm
from sklearn import metrics
from models import load_data, wdl_adult


def worker(args):
    def train(iterations, auc_enabled=True, tqdm_enabled=False):
        localiter = tqdm(range(iterations)
                         ) if tqdm_enabled else range(iterations)
        train_loss = []
        train_acc = []
        if auc_enabled:
            train_auc = []
        for it in localiter:
            loss_val, predict_y, y_val, _ = executor.run(
                'train', convert_to_numpy_ret_vals=True)
            acc_val = np.equal(
                np.argmax(y_val, 1),
                np.argmax(predict_y, 1)).astype(np.float32)
            train_loss.append(loss_val[0])
            train_acc.append(acc_val)
            if auc_enabled:
                train_auc.append(metrics.roc_auc_score(y_val, predict_y))
        if auc_enabled:
            return np.mean(train_loss), np.mean(train_acc), np.mean(train_auc)
        else:
            return np.mean(train_loss), np.mean(train_acc)

    def validate(iterations, tqdm_enabled=False):
        localiter = tqdm(range(iterations)
                         ) if tqdm_enabled else range(iterations)
        test_loss = []
        test_acc = []
        test_auc = []
        for it in localiter:
            loss_val, test_y_predicted, y_test_val = executor.run(
                'validate', convert_to_numpy_ret_vals=True)
            correct_prediction = np.equal(
                np.argmax(y_test_val, 1),
                np.argmax(test_y_predicted, 1)).astype(np.float32)
            test_loss.append(loss_val[0])
            test_acc.append(correct_prediction)
            test_auc.append(metrics.roc_auc_score(
                y_test_val, test_y_predicted))
        return np.mean(test_loss), np.mean(test_acc), np.mean(test_auc)

    batch_size = 128

    ctx = {
        'local': 'gpu:0',
        'lps': 'cpu:0,gpu:0,gpu:1,gpu:2,gpu:7',
        'lhy': 'cpu:0,gpu:1,gpu:2,gpu:3,gpu:6',
        'rps': 'cpu:0;node1:gpu:0;node1:gpu:2;node1:gpu:4;node1:gpu:6;node2:gpu:1;node2:gpu:3',
        'rhy': 'cpu:0;node1:gpu:0;node1:gpu:2;node1:gpu:4;node1:gpu:6;node2:gpu:1;node2:gpu:3'
    }[args.config]
    dense_param_ctx = {'local': 'gpu:0', 'lps': 'cpu:0,gpu:0,gpu:1,gpu:2,gpu:7', 'lhy': 'gpu:1,gpu:2,gpu:3,gpu:6',
                       'rps': 'cpu:0;node1:gpu:0;node1:gpu:2;node1:gpu:4;node1:gpu:6;node2:gpu:1;node2:gpu:3',
                       'rhy': 'node1:gpu:0;node1:gpu:2;node1:gpu:4;node1:gpu:6;node2:gpu:1;node2:gpu:3'}[args.config]
    with ht.context(ctx):
        x_train_deep, x_train_wide, y_train, x_test_deep, x_test_wide, y_test = load_data.load_adult_data()
        dense_input = [
            ht.dataloader_op([
                [x_train_deep[:, i], batch_size, 'train'],
                [x_test_deep[:, i], batch_size, 'validate'],
            ]) for i in range(12)
        ]
        sparse_input = ht.dataloader_op([
            [x_train_wide, batch_size, 'train'],
            [x_test_wide, batch_size, 'validate'],
        ])
        y_ = ht.dataloader_op([
            [y_train, batch_size, 'train'],
            [y_test, batch_size, 'validate'],
        ])
        print("Data loaded.")

        loss, prediction, y_, train_op = wdl_adult.wdl_adult(
            dense_input, sparse_input, y_, dense_param_ctx)

        eval_nodes = {'train': [loss, prediction, y_, train_op]}
        if args.val:
            print('Validation enabled...')
            eval_nodes['validate'] = [loss, prediction, y_]
        executor = ht.Executor(eval_nodes,
                               cstable_policy=args.cache, bsp=args.bsp, cache_bound=args.bound, seed=123)

    total_epoch = args.nepoch if args.nepoch > 0 else 50
    for ep in range(total_epoch):
        if ep == 5:
            start = time.time()
        print("epoch %d" % ep)
        ep_st = time.time()
        train_loss, train_acc = train(
            executor.get_batch_num('train'), auc_enabled=False)
        ep_en = time.time()
        if args.val:
            val_loss, val_acc, val_auc = validate(
                executor.get_batch_num('validate'))
            print("train_loss: %.4f, train_acc: %.4f, train_time: %.4f, test_loss: %.4f, test_acc: %.4f, test_auc: %.4f"
                  % (train_loss, train_acc, ep_en - ep_st, val_loss, val_acc, val_auc))
        else:
            print("train_loss: %.4f, train_acc: %.4f, train_time: %.4f"
                  % (train_loss, train_acc, ep_en - ep_st))
    print('all time:', time.time() - start)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='local',
                        help='[local, lps(localps), lhy(localhybrid), rps(remoteps), rhy]')
    parser.add_argument("--val", action="store_true",
                        help="whether to use validation")
    parser.add_argument("--all", action="store_true",
                        help="whether to use all data")
    parser.add_argument("--bsp", type=int, default=-1,
                        help="bsp 0, asp -1, ssp > 0")
    parser.add_argument("--cache", default=None, help="cache policy")
    parser.add_argument("--bound", default=100, help="cache bound")
    parser.add_argument("--nepoch", type=int, default=-1,
                        help="num of epochs, each train 1/10 data")
    args = parser.parse_args()
    worker(args)
