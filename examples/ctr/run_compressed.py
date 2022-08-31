import hetu as ht
import hetu.layers as htl

import os.path as osp
import numpy as np
from time import time
import argparse
from tqdm import tqdm
from sklearn import metrics


def worker(args):
    def get_auc(ground_truth_y, predicted_y):
        # auc for an epoch
        cur_gt = np.concatenate(ground_truth_y)
        cur_pr = np.concatenate(predicted_y)
        cur_gt[np.isnan(cur_gt)] = 0
        cur_gt[np.isinf(cur_gt)] = 0
        cur_pr[np.isnan(cur_pr)] = 0
        cur_pr[np.isinf(cur_pr)] = 0
        return metrics.roc_auc_score(cur_gt, cur_pr)

    def get_acc(y_val, predict_y):
        # acc for an iteration
        if y_val.shape[1] == 1:  # for criteo case
            acc_val = np.equal(
                y_val,
                predict_y > 0.5).astype(np.float32)
        else:
            acc_val = np.equal(
                np.argmax(y_val, 1),
                np.argmax(predict_y, 1)).astype(np.float32)
        return acc_val

    def train(iterations, auc_enabled=True, tqdm_enabled=False):
        localiter = tqdm(range(iterations)
                         ) if tqdm_enabled else range(iterations)
        train_loss = []
        train_acc = []
        if auc_enabled:
            ground_truth_y = []
            predicted_y = []
        for it in localiter:
            if check_auc:
                loss_val, predict_y, y_val = embed_layer.train(
                    executor, alpha_lr)[:3]
            else:
                loss_val, predict_y, y_val = executor.run(
                    'train', convert_to_numpy_ret_vals=True)[:3]
            acc_val = get_acc(y_val, predict_y)
            train_loss.append(loss_val[0])
            train_acc.append(acc_val)
            if auc_enabled:
                ground_truth_y.append(y_val)
                predicted_y.append(predict_y)
        return_vals = (np.mean(train_loss), np.mean(train_acc))
        if auc_enabled:
            train_auc = get_auc(ground_truth_y, predicted_y)
            return_vals += (train_auc,)
        return return_vals

    def validate(iterations, tqdm_enabled=False):
        localiter = tqdm(range(iterations)
                         ) if tqdm_enabled else range(iterations)
        test_loss = []
        test_acc = []
        ground_truth_y = []
        predicted_y = []
        for it in localiter:
            loss_val, test_y_predicted, y_test_val = executor.run(
                'validate', convert_to_numpy_ret_vals=True)
            correct_prediction = get_acc(y_test_val, test_y_predicted)
            test_loss.append(loss_val[0])
            test_acc.append(correct_prediction)
            ground_truth_y.append(y_test_val)
            predicted_y.append(test_y_predicted)
        test_auc = get_auc(ground_truth_y, predicted_y)
        test_loss = np.mean(test_loss)
        test_acc = np.mean(test_acc)
        nonlocal best_acc
        nonlocal best_auc
        nonlocal ep_count
        ep_count += 1
        if test_acc > best_acc:
            best_acc = test_acc
            ep_count = 0
        if test_auc > best_auc:
            best_auc = test_auc
            ep_count = 0
        return test_loss, test_acc, test_auc, ep_count >= stop_interval

    def run_epoch(train_batch_num, log_file=None):
        ep_st = time()
        train_loss, train_acc, train_auc = train(
            train_batch_num, tqdm_enabled=True)
        return_vals = (train_auc,)
        ep_en = time()
        if args.val:
            val_loss, val_acc, val_auc, early_stop = validate(
                executor.get_batch_num('validate'))
            printstr = "train_loss: %.4f, train_acc: %.4f, train_auc: %.4f, test_loss: %.4f, test_acc: %.4f, test_auc: %.4f, train_time: %.4f"\
                % (train_loss, train_acc, train_auc, val_loss, val_acc, val_auc, ep_en - ep_st)
            return_vals += (val_auc,)
        else:
            printstr = "train_loss: %.4f, train_acc: %.4f, train_auc: %.4f, train_time: %.4f"\
                % (train_loss, train_acc, train_auc, ep_en - ep_st)
        print(printstr)
        if log_file is not None:
            print(printstr, file=log_file, flush=True)
        return return_vals, early_stop

    assert args.method != 'autodim' or args.val
    batch_size = args.bs
    num_embed = 33762577
    num_embed_fields = [1460, 583, 10131227, 2202608, 305, 24, 12517, 633, 3, 93145, 5683,
                        8351593, 3194, 27, 14992, 5461306, 10, 5652, 2173, 4, 7046547, 18, 15, 286181, 105, 142572]
    num_dim = args.dim
    learning_rate = args.lr
    dataset = args.dataset

    border = np.sqrt(1 / max(num_embed_fields))
    initializer = ht.init.GenUniform(minval=-border, maxval=border)
    if args.ctx < 0:
        ctx = ht.cpu(0)
    else:
        assert args.ctx < 8
        ctx = ht.gpu(args.ctx)
    if args.ectx < 0:
        ectx = ht.cpu(0)
    else:
        assert args.ectx < 8
        ectx = ht.gpu(args.ectx)
    if args.method == 'full':
        embed_layer = htl.Embedding(
            num_embed, num_dim, initializer=initializer, ctx=ectx)
    elif args.method == 'hash':
        compress_rate = 0.5
        size_limit = None
        embed_layer = htl.HashEmbedding(
            num_embed, num_dim, compress_rate=compress_rate, size_limit=size_limit, initializer=initializer, ctx=ectx)
    elif args.method == 'compo':
        num_tables = 2
        aggregator = 'mul'
        embed_layer = htl.CompositionalEmbedding(
            num_embed, num_dim, num_tables, aggregator, initializer=initializer, ctx=ectx)
    elif args.method == 'learn':
        num_buckets = 1000000
        num_hash = 1024
        mlp_dim = 1024
        dist = 'uniform'
        embed_layer = htl.LearningEmbedding(
            num_dim, num_buckets, num_hash, mlp_dim, dist, initializer=initializer, ctx=ectx)
    elif args.method == 'dpq':
        num_choices = 32
        num_parts = 8
        share_weights = True
        mode = 'vq'
        num_slot = 26
        embed_layer = htl.DPQEmbedding(num_embed, num_dim, num_choices, num_parts,
                                       num_slot, batch_size, share_weights, mode, initializer=initializer, ctx=ectx)
    elif args.method == 'autodim':
        candidates = [2, num_dim // 4, num_dim // 2, num_dim]
        num_slot = 26
        alpha_lr = 0.001
        use_log_alpha = True
        embed_layer = htl.AutoDimEmbedding(
            num_embed, candidates, num_slot, batch_size, initializer=initializer, log_alpha=use_log_alpha, ctx=ectx)
    elif args.method == 'md':
        alpha = 0.3
        round_dim = True
        embed_layer = htl.MDEmbedding(
            num_embed_fields, num_dim, alpha, round_dim, initializer=initializer, ctx=ectx)
    elif args.method == 'prune':
        target_sparse = 0.9 * 0.444
        warm = 2
        embed_layer = htl.DeepLightEmbedding(
            num_embed, num_dim, target_sparse, warm, initializer=initializer, ctx=ectx)
    elif args.method == 'quantize':
        digit = 16
        embed_layer = htl.QuantizedEmbedding(
            num_embed, num_dim, digit, initializer=initializer, ctx=ectx)
    else:
        raise NotImplementedError

    # define models for criteo
    if args.all:
        from models.load_data import process_all_criteo_data_by_day
        func = process_all_criteo_data_by_day
    elif args.val:
        from models.load_data import process_head_criteo_data
        func = process_head_criteo_data
    else:
        from models.load_data import process_sampled_criteo_data
        func = process_sampled_criteo_data

    model = args.model(num_dim, 26, 13)

    embed_input, dense_input, y_ = embed_layer.compute_all(
        func, batch_size, args.val)
    loss, prediction = model(embed_input, dense_input, y_)
    if args.method == 'dpq' and embed_layer.mode == 'vq':
        loss = ht.add_op(loss, embed_layer.reg)
    if args.opt == 'sgd':
        optimizer = ht.optim.SGDOptimizer
    elif args.opt == 'adam':
        optimizer = ht.optim.AdamOptimizer
    elif args.opt == 'adagrad':
        optimizer = ht.optim.AdaGradOptimizer
    elif args.opt == 'amsgrad':
        optimizer = ht.optim.AMSGradOptimizer
    opt = optimizer(learning_rate=learning_rate)

    if args.method == 'autodim':
        print('Validation enabled...')
        eval_nodes = embed_layer.make_subexecutors(
            model, dense_input, y_, prediction, loss, opt)
    else:
        train_op = opt.minimize(loss)
        eval_nodes = {'train': [loss, prediction, y_, train_op]}
        if args.method == 'dpq':
            eval_nodes['train'].append(embed_layer.codebook_update)
        elif args.method == 'prune':
            eval_nodes['train'].append(embed_layer.make_prune_op())
        if args.val:
            print('Validation enabled...')
            if args.method != 'dpq':
                eval_nodes['validate'] = [loss, prediction, y_]
            else:
                val_embed_input = embed_layer.make_inference()
                val_loss, val_prediction = model(
                    val_embed_input, dense_input, y_)
                eval_nodes['validate'] = [val_loss, val_prediction, y_]
    executor_log_path = osp.join(osp.dirname(osp.abspath(__file__)), 'logs')
    executor = ht.Executor(eval_nodes, ctx=ctx, seed=123,
                           log_path=executor_log_path)

    # enable early stopping if no increase within 2 epoch
    best_acc = 0
    best_auc = 0
    stop_interval = 200
    ep_count = 0

    check_auc = False
    if args.method == 'autodim':
        executor.subexecutor['alpha'].inference = False
        executor.subexecutor['all_no_update'].inference = False
        embed_layer.get_arch_params(executor.config.placeholder_to_arr_map)
        prev_auc = None
        check_auc = True
    if args.all and dataset == 'criteo':
        print('Processing all data...')
        log_file = open(args.fname, 'w')
        total_epoch = args.nepoch if args.nepoch > 0 else 11
        train_batch_num = executor.get_batch_num('train')
        npart = 100
        base_batch_num = train_batch_num // npart
        residual = train_batch_num % npart
        for ep in range(total_epoch):
            print("epoch %d" % ep)
            results, early_stop = run_epoch(base_batch_num + (ep %
                                                              npart < residual), log_file)
            if check_auc:
                cur_auc = results[1]
                if prev_auc is not None and cur_auc <= prev_auc:
                    print("Switch to retrain stage...")
                    check_auc = False
                    executor.return_tensor_values()
                    embed_input = embed_layer.make_retrain(
                        process_all_criteo_data_by_day, num_embed_fields, executor.config.comp_stream)
                    loss, prediction = model(embed_input, dense_input, y_)
                    opt = ht.optim.AdamOptimizer(learning_rate=learning_rate)
                    train_op = opt.minimize(loss)
                    eval_nodes = {
                        'train': [loss, prediction, y_, train_op],
                        'validate': [loss, prediction, y_]}
                    executor = ht.Executor(eval_nodes, ctx=ctx, seed=123,
                                           log_path=executor_log_path)
                prev_auc = cur_auc
            if early_stop:
                print('Early stop!')
                break
    else:
        total_epoch = args.nepoch if args.nepoch > 0 else 50
        train_batch_num = executor.get_batch_num('train')
        for ep in range(total_epoch):
            print("epoch %d" % ep)
            run_epoch(train_batch_num)

    if args.method == 'prune':
        # check inference; use sparse embedding
        executor.return_tensor_values()
        val_embed_input = embed_layer.make_inference(executor)
        val_loss, val_prediction = model(
            val_embed_input, dense_input, y_)
        eval_nodes = {'validate': [val_loss, val_prediction, y_]}
        executor = ht.Executor(eval_nodes, ctx=ctx, seed=123,
                               log_path=executor_log_path)
        val_loss, val_acc, val_auc, early_stop = validate(
            executor.get_batch_num('validate'))
        printstr = "infer_loss: %.4f, infer_acc: %.4f, infer_auc: %.4f"\
            % (val_loss, val_acc, val_auc)
        print(printstr)
        if log_file is not None:
            print(printstr, file=log_file, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='DLRM_Head',
                        help="model to be tested")
    parser.add_argument("--method", type=str, default=None,
                        help="method to be used")
    parser.add_argument("--ectx", type=int, default=None,
                        help="context for embedding table")
    parser.add_argument("--ctx", type=int, default=None,
                        help="context for model")
    parser.add_argument("--bs", type=int, default=None,
                        help="batch size to be used")
    parser.add_argument("--opt", type=str, default=None,
                        help="optimizer to be used, can be SGD, Amsgrad, Adam, Adagrad")
    parser.add_argument("--dim", type=int, default=None,
                        help="dimension to be used")
    parser.add_argument("--lr", type=float, default=None,
                        help="learning rate to be used")
    parser.add_argument("--dataset", type=str, default='criteo',
                        help="dataset to be used")
    parser.add_argument("--val", action="store_true",
                        help="whether to use validation")
    parser.add_argument("--all", action="store_true",
                        help="whether to use all data")
    parser.add_argument("--nepoch", type=int, default=-1,
                        help="num of epochs, each train 1/10 data")
    args = parser.parse_args()
    args.fname = '{}'.format(args.model)
    if args.method is None:
        args.method = 'full'
    else:
        args.fname += '_{}'.format(args.method)
    if args.ctx is None:
        args.ctx = 0
    if args.ectx is None:
        args.ectx = args.ctx
    if args.bs is None:
        args.bs = 128
    else:
        args.fname += '_bs{}'.format(args.bs)
    if args.opt is None:
        args.opt = 'adam'
    else:
        args.opt = args.opt.lower()
        assert args.opt in ['sgd', 'adam', 'adagrad', 'amsgrad']
        args.fname += '_opt{}'.format(args.opt)
    if args.dim is None:
        args.dim = 16
    else:
        args.fname += '_dim{}'.format(args.dim)
    if args.lr is None:
        args.lr = 0.001
    else:
        args.fname += '_lr{}'.format(args.lr)
    args.fname += '.log'
    args.fname = osp.join(osp.dirname(
        osp.abspath(__file__)), 'logs', args.fname)
    import models
    print('Model:', args.model)
    model = eval('models.' + args.model)
    args.model = model
    worker(args)
