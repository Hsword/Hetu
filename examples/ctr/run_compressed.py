import hetu as ht
import hetu.layers as htl

import os
import os.path as osp
import numpy as np
import argparse


def worker(args):
    num_dim = args.dim
    learning_rate = args.lr
    from models.load_data import get_dataset
    dataset = get_dataset(args.dataset)()
    num_dense = dataset.num_dense
    num_sparse = dataset.num_sparse

    if args.method == 'full':
        embed_layer_type = htl.Embedding
        embedding_args = {}
    elif args.method == 'hash':
        embed_layer_type = htl.HashEmbedding
        embedding_args = {}
    elif args.method == 'compo':
        embed_layer_type = htl.CompositionalEmbedding
        embedding_args = {
            'aggregator': 'mul',  # (mul, sum)
        }
    elif args.method == 'tt':
        embed_layer_type = htl.TensorTrainEmbedding
        embedding_args = {
            'ttcore_initializer': ht.init.GenReversedTruncatedNormal(stddev=1 / ((np.sqrt(1 / 3 * max(dataset.num_embed_separate))) ** (1/3))),
        }
    elif args.method == 'dhe':
        embed_layer_type = htl.DeepHashEmbedding
        embedding_args = {
            'num_buckets': 1000000,
            'num_hash': 1024,
            'dist': 'uniform',
        }
    elif args.method == 'robe':
        embed_layer_type = htl.RobeEmbedding
        embedding_args = {
            'Z': 1,
        }
    elif args.method == 'dpq':
        embed_layer_type = htl.DPQEmbedding
        embedding_args = {
            'num_choices': 32,
            'num_parts': 8,
            'share_weights': False,
            'mode': 'vq',
        }
    elif args.method == 'mgqe':
        embed_layer_type = htl.MGQEmbedding
        embedding_args = {
            'high_num_choices': 256,
            'low_num_choices': 64,
            'num_parts': 4,
            'top_percent': 0.1,
        }
    elif args.method == 'adapt':
        embed_layer_type = htl.AdaptiveEmbedding
        embedding_args = {
            'top_percent': 0.1,
        }
    elif args.method == 'md':
        embed_layer_type = htl.MDEmbedding
        embedding_args = {
            'alpha': 0.3,
            'round_dim': True,
        }
    elif args.method == 'autodim':
        embed_layer_type = htl.AutoDimEmbedding
        embedding_args = {
            'alpha_lr': 0.001,
            'r': 1e-2,
        }
    elif args.method == 'deeplight':
        embed_layer_type = htl.DeepLightEmbedding
        embedding_args = {
            'warm': 2,
        }
    elif args.method == 'pep':
        embed_layer_type = htl.PEPEmbedding
        embedding_args = {
            'threshold_type': 'feature_dimension',
            'threshold_init': -150,
        }
    elif args.method == 'autosrh':
        embed_layer_type = htl.AutoSrhEmbedding
        embedding_args = {
            'nsplit': 10,
            'warm_start_epochs': 1,
            'alpha_l1': 0.00001,
            'alpha_lr': 0.001,
        }
    elif args.method == 'quantize':
        embed_layer_type = htl.QuantizedEmbedding
        embedding_args = {
            'digit': 16,
            'scale': 0.01,
            'middle': 0,
            'use_qparam': False,
        }
    elif args.method == 'alpt':
        embed_layer_type = htl.ALPTEmbedding
        embedding_args = {
            'digit': 16,
            'init_scale': 0.01,
        }
    else:
        raise NotImplementedError

    args.embedding_args = embedding_args

    # define models
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

    trainer = ht.sched.get_trainer(embed_layer_type)(
        dataset, model, opt, args)

    if args.phase == 'train':
        trainer.fit()
    else:
        trainer.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='dlrm',
                        help="model to be tested")
    parser.add_argument("--method", type=str, default='full',
                        help="method to be used",
                        choices=['full', 'hash', 'compo', 'tt',
                                 'dhe', 'robe', 'dpq', 'mgqe', 'adapt',
                                 'md', 'autodim',
                                 'deeplight', 'pep', 'autosrh',
                                 'quantize', 'alpt',])
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
                        help="dataset to be used", choices=['criteo', 'avazu'])
    parser.add_argument("--nepoch", type=float, default=0.1,
                        help="num of epochs")
    parser.add_argument("--num_test_every_epoch", type=int, default=100,
                        help="evaluate each 1/100 epoch in default")
    parser.add_argument("--seed", type=int, default=123,
                        help="random seed")
    parser.add_argument("--separate_fields", type=int, default=None,
                        help="whether seperate fields")
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

    if args.method in ('robe', 'deeplight', 'pep', 'autosrh', 'quantize', 'alpt', 'autodim'):
        # autodim not use multi in the first stage, use multi in the second stage.
        args.use_multi = 0
    elif args.method in ('compo', 'md', 'tt', 'dhe', 'mgqe', 'adapt'):
        # dhe, mgqe, adapt both is ok; use multi is better according to semantic meaning.
        args.use_multi = 1
    if args.method == 'autodim' and args.phase == 'test':
        args.use_multi = 1
    if args.method == 'robe':
        # robe use multi, separate fields controls whether using slot coefficient
        if args.separate_fields is None:
            args.separate_fields = args.use_multi
    else:
        args.separate_fields = args.use_multi

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
