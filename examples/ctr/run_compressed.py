import hetu as ht
import hetu.layers as htl

import os
import os.path as osp
import numpy as np
import argparse


def get_ctx(idx):
    if idx < 0:
        ctx = ht.cpu(0)
    else:
        assert idx < 8
        ctx = ht.gpu(idx)
    return ctx


def worker(args):

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
        embed_layer = htl.AutoDimEmbedding(
            num_embed_fields, candidates, num_slot, batch_size, alpha_lr, initializer=initializer, ctx=ectx)
    elif args.method == 'md':
        alpha = 0.3
        round_dim = True
        embed_layer = htl.MDEmbedding(
            num_embed_fields, num_dim, alpha, round_dim, initializer=initializer, ctx=ectx)
    elif args.method == 'prune':
        target_sparse = args.compress_rate
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

    if args.opt == 'sgd':
        optimizer = ht.optim.SGDOptimizer
    elif args.opt == 'adam':
        optimizer = ht.optim.AdamOptimizer
    elif args.opt == 'adagrad':
        optimizer = ht.optim.AdaGradOptimizer
    elif args.opt == 'amsgrad':
        optimizer = ht.optim.AMSGradOptimizer
    opt = optimizer(learning_rate=learning_rate)

    from models.load_data import process_all_criteo_data_by_day
    trainer = ht.sched.get_trainer(embed_layer)(
        embed_layer, process_all_criteo_data_by_day, model, opt, args)

    if args.phase == 'train':
        if dataset == 'criteo':
            trainer.fit()
        else:
            raise NotImplementedError
    else:
        trainer.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='dlrm',
                        help="model to be tested")
    parser.add_argument("--method", type=str, default='full',
                        help="method to be used",
                        choices=['full', 'robe', 'hash', 'compo',
                                 'learn', 'dpq', 'autodim', 'md',
                                 'prune', 'quantize'])
    parser.add_argument("--phase", type=str, default='train',
                        help='train or test',
                        choices=['train', 'test'])
    parser.add_argument("--ectx", type=int, default=None,
                        help="context for embedding table")
    parser.add_argument("--ctx", type=int, default=0,
                        help="context for model")
    parser.add_argument("--bs", type=int, default=128,
                        help="batch size to be used")
    parser.add_argument("--opt", type=str, default='sgd',
                        help="optimizer to be used",
                        choices=['sgd', 'adam', 'adagrad', 'amsgrad'])
    parser.add_argument("--dim", type=int, default=16,
                        help="dimension to be used")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate to be used")
    parser.add_argument("--dataset", type=str, default='criteo',
                        help="dataset to be used")
    parser.add_argument("--nepoch", type=float, default=0.1,
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
    if args.ectx is None:
        args.ectx = args.ctx

    if args.method in ('robe', 'compo', 'learn', 'dpq', 'prune', 'quantize', 'autodim'):
        # TODO: improve in future
        # autodim not use multi in the first stage, use multi in the second stage.
        args.use_multi = 0
    elif args.method in ('md'):
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
    args.result_file = '_'.join(infos) + '.log'
    if args.phase == 'test':
        args.result_file = args.phase + args.result_file
    args.log_dir = osp.join(osp.dirname(osp.abspath(__file__)), 'logs')
    os.makedirs(args.log_dir, exist_ok=True)
    args.save_dir = osp.join(osp.dirname(
        osp.abspath(__file__)), 'ckpts', args.result_file[:-4])
    args.result_file = osp.join(args.log_dir, args.result_file)
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
