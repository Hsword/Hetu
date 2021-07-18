import hetu as ht
from hetu.launcher import launch

import os
import os.path as osp
import numpy as np
import yaml
import time
import argparse
from tqdm import tqdm
from sklearn import metrics


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
            if y_val.shape[1] == 1:  # for criteo case
                acc_val = np.equal(
                    y_val,
                    predict_y > 0.5).astype(np.float32)
            else:
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
            if y_test_val.shape[1] == 1:  # for criteo case
                correct_prediction = np.equal(
                    y_test_val,
                    test_y_predicted > 0.5).astype(np.float32)
            else:
                correct_prediction = np.equal(
                    np.argmax(y_test_val, 1),
                    np.argmax(test_y_predicted, 1)).astype(np.float32)
            test_loss.append(loss_val[0])
            test_acc.append(correct_prediction)
            test_auc.append(metrics.roc_auc_score(
                y_test_val, test_y_predicted))
        return np.mean(test_loss), np.mean(test_acc), np.mean(test_auc)

    def get_current_shard(data):
        if args.comm is not None:
            part_size = data.shape[0] // nrank
            start = part_size * rank
            end = start + part_size if rank != nrank - 1 else data.shape[0]
            return data[start:end]
        else:
            return data

    batch_size = 128
    dataset = args.dataset
    model = args.model
    device_id = 0

    if args.comm == 'PS':
        rank = ht.get_worker_communicate().rank()
        nrank = int(os.environ['DMLC_NUM_WORKER'])
        device_id = rank % 8
    elif args.comm == 'Hybrid':
        comm = ht.wrapped_mpi_nccl_init()
        device_id = comm.dev_id
        rank = comm.rank
        nrank = int(os.environ['DMLC_NUM_WORKER'])

    if dataset == 'criteo':
        # define models for criteo
        if args.all:
            from models.load_data import process_all_criteo_data
            dense, sparse, labels = process_all_criteo_data(
                return_val=args.val)
        elif args.val:
            from models.load_data import process_head_criteo_data
            dense, sparse, labels = process_head_criteo_data(return_val=True)
        else:
            from models.load_data import process_sampled_criteo_data
            dense, sparse, labels = process_sampled_criteo_data()
        if isinstance(dense, tuple):
            dense_input = ht.dataloader_op([[get_current_shard(dense[0]), batch_size, 'train'], [
                                           get_current_shard(dense[1]), batch_size, 'validate']])
            sparse_input = ht.dataloader_op([[get_current_shard(sparse[0]), batch_size, 'train'], [
                                            get_current_shard(sparse[1]), batch_size, 'validate']])
            y_ = ht.dataloader_op([[get_current_shard(labels[0]), batch_size, 'train'], [
                                  get_current_shard(labels[1]), batch_size, 'validate']])
        else:
            dense_input = ht.dataloader_op(
                [[get_current_shard(dense), batch_size, 'train']])
            sparse_input = ht.dataloader_op(
                [[get_current_shard(sparse), batch_size, 'train']])
            y_ = ht.dataloader_op(
                [[get_current_shard(labels), batch_size, 'train']])
    elif dataset == 'adult':
        from models.load_data import load_adult_data
        x_train_deep, x_train_wide, y_train, x_test_deep, x_test_wide, y_test = load_adult_data()
        dense_input = [
            ht.dataloader_op([
                [get_current_shard(x_train_deep[:, i]), batch_size, 'train'],
                [get_current_shard(x_test_deep[:, i]), batch_size, 'validate'],
            ]) for i in range(12)
        ]
        sparse_input = ht.dataloader_op([
            [get_current_shard(x_train_wide), batch_size, 'train'],
            [get_current_shard(x_test_wide), batch_size, 'validate'],
        ])
        y_ = ht.dataloader_op([
            [get_current_shard(y_train), batch_size, 'train'],
            [get_current_shard(y_test), batch_size, 'validate'],
        ])
    else:
        raise NotImplementedError
    print("Data loaded.")

    loss, prediction, y_, train_op = model(dense_input, sparse_input, y_)

    eval_nodes = {'train': [loss, prediction, y_, train_op]}
    if args.val:
        print('Validation enabled...')
        eval_nodes['validate'] = [loss, prediction, y_]
    executor_log_path = osp.join(osp.dirname(osp.abspath(__file__)), 'logs')
    executor = ht.Executor(eval_nodes, ctx=ht.gpu(device_id),
                           comm_mode=args.comm, cstable_policy=args.cache, bsp=args.bsp, cache_bound=args.bound, seed=123, log_path=executor_log_path)

    if args.all and dataset == 'criteo':
        print('Processing all data...')
        file_path = '%s_%s' % ({None: 'local', 'PS': 'ps', 'Hybrid': 'hybrid'}[
                               args.comm], args.raw_model)
        file_path += '%d.log' % rank if args.comm else '.log'
        file_path = osp.join(osp.dirname(
            osp.abspath(__file__)), 'logs', file_path)
        log_file = open(file_path, 'w')
        total_epoch = args.nepoch if args.nepoch > 0 else 11
        for ep in range(total_epoch):
            print("ep: %d" % ep)
            ep_st = time.time()
            train_loss, train_acc, train_auc = train(executor.get_batch_num(
                'train') // 10 + (ep % 10 == 9) * (executor.get_batch_num('train') % 10), tqdm_enabled=True)
            ep_en = time.time()
            if args.val:
                val_loss, val_acc, val_auc = validate(
                    executor.get_batch_num('validate'))
                printstr = "train_loss: %.4f, train_acc: %.4f, train_auc: %.4f, test_loss: %.4f, test_acc: %.4f, test_auc: %.4f, train_time: %.4f"\
                    % (train_loss, train_acc, train_auc, val_loss, val_acc, val_auc, ep_en - ep_st)
            else:
                printstr = "train_loss: %.4f, train_acc: %.4f, train_auc: %.4f, train_time: %.4f"\
                    % (train_loss, train_acc, train_auc, ep_en - ep_st)
            print(printstr)
            log_file.write(printstr + '\n')
            log_file.flush()
    else:
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
    parser.add_argument("--model", type=str, required=True,
                        help="model to be tested")
    parser.add_argument("--val", action="store_true",
                        help="whether to use validation")
    parser.add_argument("--all", action="store_true",
                        help="whether to use all data")
    parser.add_argument("--comm", default=None,
                        help="whether to use distributed setting, can be None, AllReduce, PS, Hybrid")
    parser.add_argument("--bsp", action="store_true",
                        help="whether to use bsp instead of asp")
    parser.add_argument("--cache", default=None, help="cache policy")
    parser.add_argument("--bound", default=100, help="cache bound")
    parser.add_argument("--config", type=str, default=osp.join(osp.dirname(
        osp.abspath(__file__)), "./settings/local_s1_w4.yml"), help="configuration for ps")
    parser.add_argument("--nepoch", type=int, default=-1,
                        help="num of epochs, each train 1/10 data")
    args = parser.parse_args()
    import models
    print('Model:', args.model)
    model = eval('models.' + args.model)
    args.dataset = args.model.split('_')[-1]
    args.raw_model = args.model
    args.model = model
    if args.comm is None:
        worker(args)
    elif args.comm == 'Hybrid':
        settings = yaml.load(open(args.config).read(), Loader=yaml.FullLoader)
        value = settings['shared']
        os.environ['DMLC_ROLE'] = 'worker'
        for k, v in value.items():
            os.environ[k] = str(v)
        worker(args)
    elif args.comm == 'PS':
        launch(worker, args)
    else:
        raise NotImplementedError
