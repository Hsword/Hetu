import hetu as ht
import hetu.layers as htl

import os
import os.path as osp
import pickle
import numpy as np
from time import time
import argparse
from tqdm import tqdm
from sklearn import metrics


def get_data(args):
    batch_size = args.bs

    from models.load_data import process_all_criteo_data_by_day
    dense, sparse, labels = process_all_criteo_data_by_day(
        return_val=True, separate_fields=args.use_multi)

    tr_name = 'train'
    va_name = 'validate'

    def make_dataloader_op(tr_data, va_data, dtype=np.float32):
        train_dataloader = ht.Dataloader(
            tr_data, batch_size, tr_name, dtype=dtype)
        valid_dataloader = ht.Dataloader(
            va_data, batch_size, va_name, dtype=dtype)
        data_op = ht.dataloader_op(
            [train_dataloader, valid_dataloader], dtype=dtype)
        return data_op

    # define models for criteo
    tr_dense, va_dense = dense
    tr_sparse, va_sparse = sparse
    tr_labels, va_labels = labels
    tr_labels = tr_labels.reshape((-1, 1))
    va_labels = va_labels.reshape((-1, 1))
    dense_input = make_dataloader_op(tr_dense, va_dense)
    y_ = make_dataloader_op(tr_labels, va_labels)
    if args.use_multi:
        new_sparse_ops = []
        for i in range(tr_sparse.shape[1]):
            cur_data = make_dataloader_op(
                tr_sparse[:, i], None if va_sparse is None else va_sparse[:, i], dtype=np.int32)
            new_sparse_ops.append(cur_data)
        embed_input = new_sparse_ops
    else:
        embed_input = make_dataloader_op(tr_sparse, va_sparse, dtype=np.int32)
    print("Data loaded.")
    return embed_input, dense_input, y_


def get_ctx(idx):
    if idx < 0:
        ctx = ht.cpu(0)
    else:
        assert idx < 8
        ctx = ht.gpu(idx)
    return ctx


def handle_inf_nan(arr):
    arr[np.isnan(arr)] = 0
    arr[np.isinf(arr)] = 0
    return arr


def get_auc(ground_truth_y, predicted_y):
    # auc for an epoch
    cur_gt = np.concatenate(ground_truth_y)
    cur_pr = np.concatenate(predicted_y)
    cur_gt = handle_inf_nan(cur_gt)
    cur_pr = handle_inf_nan(cur_pr)
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


def worker(args):

    def train(iterations, epoch, part, auc_enabled=True, tqdm_enabled=False):
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
            executor.multi_log(
                {'epoch': epoch, 'part': part, 'train_loss': loss_val})
            executor.step_logger()
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

    def validate(iterations, epoch, part, tqdm_enabled=False):
        localiter = tqdm(range(iterations)
                         ) if tqdm_enabled else range(iterations)
        test_loss = []
        test_acc = []
        ground_truth_y = []
        predicted_y = []
        for it in localiter:
            loss_value, test_y_predicted, y_test_value = executor.run(
                'validate', convert_to_numpy_ret_vals=True)
            correct_prediction = get_acc(y_test_value, test_y_predicted)
            test_loss.append(loss_value[0])
            test_acc.append(correct_prediction)
            ground_truth_y.append(y_test_value)
            predicted_y.append(test_y_predicted)
        test_auc = get_auc(ground_truth_y, predicted_y)
        test_loss = np.mean(test_loss)
        test_acc = np.mean(test_acc)
        nonlocal best_acc
        nonlocal best_auc
        nonlocal ep_count
        ep_count += 1
        # if test_acc > best_acc:
        #     best_acc = test_acc
        #     ep_count = 0
        if test_auc > best_auc:
            best_auc = test_auc
            ep_count = 0
        try_save_ckpt(test_auc, (epoch, part))
        return test_loss, test_acc, test_auc, ep_count >= stop_interval

    def try_save_ckpt(test_auc, cur_meta):
        if args.save_topk > 0 and test_auc > topk_auc[-1]:
            idx = None
            for i, auc in enumerate(topk_auc):
                if test_auc >= auc:
                    idx = i
                    break
            if idx is not None:
                topk_auc.insert(idx, test_auc)
                topk_ckpts.insert(idx, cur_meta)
                ep, part = cur_meta
                executor.save(args.save_dir, f'ep{ep}_{part}.pkl', {
                              'epoch': ep, 'part': part, 'npart': args.num_test_every_epoch})
                rm_auc = topk_auc.pop()
                rm_meta = topk_ckpts.pop()
                print(
                    f'Save ep{ep}_{part}.pkl with auc {test_auc}; current ckpts {topk_ckpts} with aucs {topk_auc}.')
                if rm_meta is not None:
                    ep, part = rm_meta
                    os.remove(osp.join(args.save_dir, f'ep{ep}_{part}.pkl'))
                    print(f'Remove ep{ep}_{part}.pkl with auc {rm_auc}.')

    def run_epoch(train_batch_num, epoch, part, log_file=None):
        ep_st = time()
        train_loss, train_acc, train_auc = train(
            train_batch_num, epoch, part, tqdm_enabled=True)
        return_vals = (train_auc,)
        results = {'epoch': epoch, 'part': part, 'avg_train_loss': train_loss,
                   'train_acc': train_acc, 'train_auc': train_auc}
        ep_en = time()
        if args.val:
            test_loss, test_acc, test_auc, early_stop = validate(
                executor.get_batch_num('validate'), epoch, part, True)
            printstr = "train_loss: %.4f, train_acc: %.4f, train_auc: %.4f, test_loss: %.4f, test_acc: %.4f, test_auc: %.4f, train_time: %.4f"\
                % (train_loss, train_acc, train_auc, test_loss, test_acc, test_auc, ep_en - ep_st)
            return_vals += (test_auc,)
            results.update({'avg_test_loss': test_loss,
                            'test_acc': test_acc, 'test_auc': test_auc})
        else:
            printstr = "train_loss: %.4f, train_acc: %.4f, train_auc: %.4f, train_time: %.4f"\
                % (train_loss, train_acc, train_auc, ep_en - ep_st)
        executor.multi_log(results)
        executor.step_logger()
        print(printstr)
        if log_file is not None:
            print(printstr, file=log_file, flush=True)
        return return_vals, early_stop

    topk_auc = [0 for _ in range(args.save_topk)]
    topk_ckpts = [None for _ in range(args.save_topk)]

    assert args.method != 'autodim' or args.val
    batch_size = args.bs
    num_dim = args.dim
    learning_rate = args.lr
    dataset = args.dataset
    if args.dataset == 'criteo':
        num_embed = 33762577
        num_embed_fields = [1460, 583, 10131227, 2202608, 305, 24, 12517, 633, 3, 93145, 5683,
                            8351593, 3194, 27, 14992, 5461306, 10, 5652, 2173, 4, 7046547, 18, 15, 286181, 105, 142572]
    else:
        raise NotImplementedError

    if args.debug:
        print('Use zero initializer for debug.')
        initializer = ht.init.GenZeros()
    else:
        border = np.sqrt(1 / max(num_embed_fields))
        initializer = ht.init.GenUniform(minval=-border, maxval=border)
    ctx = get_ctx(args.ctx)
    ectx = get_ctx(args.ectx)

    if args.method == 'full':
        if args.use_multi:
            embed_layer = htl.MultipleEmbedding(
                num_embed_fields, num_dim, initializer=initializer, ctx=ectx)
        else:
            embed_layer = htl.Embedding(
                num_embed, num_dim, initializer=initializer, ctx=ectx)
    elif args.method == 'robe':
        compress_rate = args.compress_rate
        size_limit = None
        Z = 1
        embed_layer = htl.RobeEmbedding(
            num_embed, num_dim, compress_rate=compress_rate, size_limit=size_limit, Z=Z, initializer=initializer, ctx=ectx)
    elif args.method == 'hash':
        compress_rate = args.compress_rate
        size_limit = None
        if args.use_multi:
            embed_layer = htl.MultipleHashEmbedding(
                num_embed_fields, num_dim, compress_rate=compress_rate, size_limit=size_limit, initializer=initializer, ctx=ectx)
        else:
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
        scale = 0.01
        middle = 0
        use_qparam = False
        embed_layer = htl.QuantizedEmbedding(
            num_embed, num_dim, digit, scale=scale, middle=middle, use_qparam=use_qparam, initializer=initializer, ctx=ectx)
    else:
        raise NotImplementedError

    # define models
    if args.dataset == 'criteo':
        num_sparse = 26
        num_dense = 13
    else:
        raise NotImplementedError
    model = args.model(num_dim, num_sparse, num_dense)

    data_ops = get_data(args)
    embed_input, dense_input, y_ = data_ops
    data_ops = data_ops[0] + list(data_ops[1:]) if args.use_multi else data_ops
    loss, prediction = model(embed_layer(embed_input), dense_input, y_)

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
            if args.method != 'dpq':
                eval_nodes['validate'] = [loss, prediction, y_]
            else:
                test_embed_input = embed_layer.make_inference()
                test_loss, test_prediction = model(
                    test_embed_input, dense_input, y_)
                eval_nodes['validate'] = [test_loss, test_prediction, y_]
    project = 'embedmem'
    run_name = osp.split(args.fname)[1][:-4]
    executor = ht.Executor(
        eval_nodes,
        ctx=ctx,
        seed=args.seed,
        log_path=args.log_dir,
        logger=args.logger,
        project=project,
        run_name=run_name,
        run_id=args.run_id,
    )
    start_ep = 0
    total_epoch = args.nepoch * args.num_test_every_epoch if args.nepoch > 0 else 11
    train_batch_num = executor.get_batch_num('train')
    npart = args.num_test_every_epoch
    base_batch_num = train_batch_num // npart
    residual = train_batch_num % npart
    if args.load_ckpt is not None:
        with open(args.load_ckpt, 'rb') as fr:
            meta = pickle.load(fr)
            executor.load_dict(meta['state_dict'])
            executor.load_seeds(meta['seed'])
            start_epoch = meta['epoch']
            start_part = meta['part'] + 1
            assert meta['npart'] == args.num_test_every_epoch
            start_ep = start_epoch * args.num_test_every_epoch + start_part
            for op in data_ops:
                op.set_batch_index('train', start_part * base_batch_num)
            print(f'Load ckpt from {osp.split(args.load_ckpt)[-1]}.')
    executor.set_config(args)

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
    if dataset == 'criteo':
        log_file = open(args.fname, 'w')
        for ep in range(start_ep, total_epoch):
            real_ep = ep // npart
            real_part = ep % npart
            print(f"Epoch {real_ep}({real_part})")
            results, early_stop = run_epoch(
                base_batch_num + (real_part < residual), real_ep, real_part, log_file)
            if check_auc:
                cur_auc = results[1]
                if prev_auc is not None and cur_auc <= prev_auc:
                    print("Switch to retrain stage.")
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
                    executor = ht.Executor(eval_nodes, ctx=ctx, seed=args.seed,
                                           log_path=args.log_dir)
                prev_auc = cur_auc
            if early_stop:
                print('Early stop!')
                break
    else:
        total_epoch = args.nepoch if args.nepoch > 0 else 50
        train_batch_num = executor.get_batch_num('train')
        for ep in range(total_epoch):
            print(f"epoch {ep}")
            run_epoch(train_batch_num, ep, 0)

    if args.method == 'prune':
        # check inference; use sparse embedding
        executor.return_tensor_values()
        test_embed_input = embed_layer.make_inference(executor)
        test_loss, test_prediction = model(
            test_embed_input, dense_input, y_)
        eval_nodes = {'validate': [test_loss, test_prediction, y_]}
        executor = ht.Executor(eval_nodes, ctx=ctx,
                               seed=args.seed, log_path=args.log_dir)
        test_loss, test_acc, test_auc, early_stop = validate(
            executor.get_batch_num('validate'))
        printstr = "infer_loss: %.4f, infer_acc: %.4f, infer_auc: %.4f"\
            % (test_loss, test_acc, test_auc)
        print(printstr)
        if log_file is not None:
            print(printstr, file=log_file, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='dlrm',
                        help="model to be tested")
    parser.add_argument("--method", type=str, default='full',
                        help="method to be used")
    parser.add_argument("--ectx", type=int, default=None,
                        help="context for embedding table")
    parser.add_argument("--ctx", type=int, default=0,
                        help="context for model")
    parser.add_argument("--bs", type=int, default=128,
                        help="batch size to be used")
    parser.add_argument("--opt", type=str, default='sgd',
                        help="optimizer to be used, can be SGD, Amsgrad, Adam, Adagrad")
    parser.add_argument("--dim", type=int, default=16,
                        help="dimension to be used")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate to be used")
    parser.add_argument("--dataset", type=str, default='criteo',
                        help="dataset to be used")
    # parser.add_argument("--val", action="store_true",
    #                     help="whether to use validation")
    parser.add_argument("--nepoch", type=int, default=-1,
                        help="num of epochs")
    parser.add_argument("--num_test_every_epoch", type=int, default=100,
                        help="evaluate each 1/100 epoch in default")
    parser.add_argument("--seed", type=int, default=123,
                        help="random seed")
    parser.add_argument("--use_multi", type=int, default=0,
                        help="whether use multi embedding")
    parser.add_argument("--compress_rate", type=float, default=0.5,
                        help="compress rate")
    parser.add_argument("--logger", type=str, default="hetu",
                        help="logger to be used")
    parser.add_argument("--run_id", type=str, default=None,
                        help="run id to be logged")
    parser.add_argument("--debug", action="store_true",
                        help="whether in debug mode")
    parser.add_argument("--load_ckpt", type=str, default=None,
                        help="ckpt to be used")
    parser.add_argument("--save_topk", type=int, default=0,
                        help="number of ckpts to be saved")
    args = parser.parse_args()

    args.opt = args.opt.lower()
    assert args.opt in ['sgd', 'adam', 'adagrad', 'amsgrad']
    if args.ectx is None:
        args.ectx = args.ctx
    args.val = True

    assert args.method in ('full', 'robe', 'hash', 'compo',
                           'learn', 'dpq', 'autodim', 'md', 'prune', 'quantize')
    if args.method in ('robe', 'compo', 'learn', 'dpq', 'prune', 'quantize'):
        # TODO: improve in future
        args.use_multi = 0
    elif args.method in ('md', 'autodim'):
        args.use_multi = 1

    infos = [
        f'{args.model}',
        f'{args.dataset}',
        f'{args.method}',
        f'{args.opt}',
        f'dim{args.dim}',
        # f'bs{args.bs}',
        # f'lr{args.lr}',
        f'cr{args.compress_rate}',
        f'multi{args.use_multi}',
    ]
    if args.debug:
        infos.append(f'debug')
    args.fname = '_'.join(infos) + '.log'
    args.log_dir = osp.join(osp.dirname(osp.abspath(__file__)), 'logs')
    os.makedirs(args.log_dir, exist_ok=True)
    args.save_dir = osp.join(osp.dirname(
        osp.abspath(__file__)), 'ckpts', args.fname[:-4])
    args.fname = osp.join(args.log_dir, args.fname)
    if args.save_topk > 0:
        if osp.isdir(args.save_dir):
            print('Warning: the save dir already exists!')
        os.makedirs(args.save_dir, exist_ok=True)
    if args.load_ckpt is not None:
        if not osp.isfile(args.load_ckpt):
            args.load_ckpt = osp.join(args.save_dir, args.load_ckpt)
            assert osp.isfile(args.load_ckpt)

    print(f'Use {args.model} on {args.dataset}.')
    if args.model.lower().startswith('dlrm'):
        from models import DLRM_Head
        model = DLRM_Head
    else:
        raise NotImplementedError
    args.model = model
    worker(args)
