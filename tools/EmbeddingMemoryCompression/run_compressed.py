import hetu as ht

import os
import os.path as osp
import argparse


def worker(args):
    num_dim = args.dim
    learning_rate = args.lr
    from models.load_data import get_dataset
    dataset = get_dataset(args.dataset)(path=args.data_path)
    num_dense = dataset.num_dense
    num_sparse = dataset.num_sparse

    if args.method == 'full':
        embedding_args = {}
    elif args.method == 'hash':
        embedding_args = {
            'threshold': args.threshold,
        }
    elif args.method == 'compo':
        embedding_args = {
            'aggregator': 'mul',  # (mul, sum)
            'threshold': args.threshold,
        }
    elif args.method == 'tt':
        embedding_args = {
            'threshold': args.threshold,
        }
    elif args.method == 'dhe':
        embedding_args = {
            'num_buckets': 1000000,
            'num_hash': 1024,
            'dist': 'uniform',
            'threshold': args.threshold,
        }
    elif args.method == 'robe':
        embedding_args = {
            'Z': 1,
        }
    elif args.method == 'dpq':
        embedding_args = {
            'num_choices': 32,
            'num_parts': 8,
            'share_weights': False,
            'mode': 'vq',
            'threshold': args.threshold,
        }
    elif args.method == 'mgqe':
        embedding_args = {
            'high_num_choices': 256,
            'low_num_choices': 64,
            'num_parts': 4,
            'top_percent': 0.1,
            'threshold': args.threshold,
        }
    elif args.method == 'adapt':
        embedding_args = {
            'top_percent': 0.1,
            'exact_split': 0,
            'threshold': args.threshold,
        }
    elif args.method == 'dedup':
        embedding_args = {
            'stage': args.stage,
            'fix_emb': False, # try: fix_emb True or False; use_multi 0 or 1
            'block_cap': 10000 * args.dim,
            'lsh_threshold': 0, # increase for larger progress; adjust (2,3,4,5,...,until num_l=90; normally smaller than 10)
            'fp': 0.01, # if AUC bad, decrease this value (*= 0.1; > 0)
            'sim': 0.7, # if AUC bad, increase this value (+= 0.1; < 1)
        }
    elif args.method == 'md':
        embedding_args = {
            'round_dim': True,
        }
    elif args.method == 'autodim':
        embedding_args = {
            'stage': args.stage,
            'alpha_lr': 0.001,
            'r': 1e-2,
            'reset_retrain': 0,
            'ignore_second': 1,  # 0 or 1
        }
    elif args.method == 'optembed':
        embedding_args = {
            'stage': args.stage,
            'alpha': 1e-5,  # the coef of regularization; 1e-6 or 1e-5 or 1e-4 for criteo and avazu
            'thresh_lr': 1e-4,  # lr for threshold, 1e-4 or same scale with lr
            'keep_num': 0,
            'mutation_num': 10,
            'crossover_num': 10,
            'm_prob': 0.1,
            'nepoch_search': 30,
        }
    elif args.method == 'deeplight':
        embedding_args = {
            'stage': args.stage,
            # candidates: (feature_dim, feature, dim), most to least buffer size
            'buffer_conf': 'feature_dim',
            'stop_deviation': 1e-3,
        }
    elif args.method == 'pep':
        embedding_args = {
            'threshold_type': 'feature_dimension',
            'threshold_init': -150,
        }
    elif args.method == 'autosrh':
        embedding_args = {
            'nsplit': 6,
            'stage': args.stage,
            'alpha_l1': 0.00001,
            'alpha_lr': 0.001,
        }
    elif args.method == 'quantize':
        embedding_args = {
            'digit': 16,
            'scale': 0.01,
            'middle': 0,
            'use_qparam': False,
        }
    elif args.method == 'alpt':
        embedding_args = {
            'digit': 16,
            'init_scale': 0.01,
            'scale_lr': 2e-5,
        }
    else:
        raise NotImplementedError

    args.embedding_args = embedding_args

    embedding_details=""
    for k,v in args.embedding_args.items():
        embedding_details+=str(k)
        embedding_details+=str(v)
        embedding_details+='_'
    embedding_details = embedding_details[0:-1]+".log"
    args.result_file = args.result_file[0:-4]+embedding_details

    args.save_dir = osp.join(args.save_dir, args.result_file[:-4])
    args.result_file = osp.join(args.log_dir, args.result_file)
    if args.save_topk > 0:
        if osp.isdir(args.save_dir):
            print('Warning: the save dir already exists!')
        os.makedirs(args.save_dir, exist_ok=True)
    
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

    from methods import get_trainer
    trainer = get_trainer(args.method)(dataset, model, opt, args)

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
                        choices=['full', 'hash', 'compo', 'tt', 'dhe',
                                 'robe', 'dpq', 'mgqe', 'adapt', 'dedup',
                                 'md', 'autodim', 'optembed',
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
                        help="dataset to be used", choices=[
                            'criteo', 'avazu', 'criteotb', 'criteo2core',
                            'sparsified', 'densified', 'moreskewed', 'lessskewed',
                            'avazu2core', 'avazusparsified', 'avazudensified',
                            'avazumoreskewed', 'avazulessskewed', 'company',
                        ])
    parser.add_argument("--data_path", type=str, default=None,
                        help="path to dataset")
    parser.add_argument("--nepoch", type=float, default=0.1,
                        help="num of epochs")
    parser.add_argument("--num_test_every_epoch", type=int, default=10,
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
    parser.add_argument("--threshold", type=int, default=0,
                        help="the threshold for inter-feature compression")
    parser.add_argument("--stage", type=int, default=1,
                        help="the start stage for train/test")
    parser.add_argument("--load_ckpt", type=str, default=None,
                        help="ckpt to be used")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="directory for logging")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="directory for saving")
    parser.add_argument("--save_topk", type=int, default=0,
                        help="number of ckpts to be saved")
    parser.add_argument("--check_val", type=int, default=0,
                        help="whether check validation data during training", choices=[0, 1])
    parser.add_argument("--check_test", type=int, default=1,
                        help="whether check test data during training", choices=[0, 1])
    parser.add_argument("--early_stop_steps", type=int, default=10,
                        help="early stopping if no improvement over steps")
    args = parser.parse_args()

    args.opt = args.opt.lower()
    if args.ectx is None:
        args.ectx = args.ctx

    if args.method in ('robe', 'deeplight', 'pep', 'autosrh', 'quantize', 'alpt', 'autodim', 'optembed'):
        # autodim not use multi in the first stage, use multi in the second stage.
        args.use_multi = 0
    elif args.method in ('compo', 'md', 'tt', 'dhe', 'mgqe', 'adapt'):
        # dhe, mgqe, adapt both is ok; use multi is better according to semantic meaning.
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
        f'bs{args.bs}',
        f'lr{args.lr}',
        f'cr{args.compress_rate}',
        f'multi{args.use_multi}',
    ]
    if args.debug:
        infos.append(f'debug')
    args.result_file = '_'.join(infos) + '.log'
    if args.phase == 'test':
        args.result_file = args.phase + args.result_file
    if args.log_dir is None:
        args.log_dir = osp.join(osp.dirname(osp.abspath(__file__)), 'logs')
    os.makedirs(args.log_dir, exist_ok=True)
    if args.save_dir is None:
        args.save_dir = osp.join(osp.dirname(osp.abspath(__file__)), 'ckpts')
    '''args.save_dir = osp.join(args.save_dir, args.result_file[:-4])
    args.result_file = osp.join(args.log_dir, args.result_file)
    if args.save_topk > 0:
        if osp.isdir(args.save_dir):
            print('Warning: the save dir already exists!')
        os.makedirs(args.save_dir, exist_ok=True)'''
    if args.load_ckpt is not None:
        if not osp.isfile(args.load_ckpt):
            args.load_ckpt = osp.join(args.save_dir, args.load_ckpt)
            assert osp.isfile(args.load_ckpt)

    print(f'Use {args.model} on {args.dataset}.')
    model_name = args.model.lower()
    if model_name.startswith('dlrm'):
        from models import DLRM_Head
        model = DLRM_Head
    elif model_name.startswith('wdl'):
        from models import WDL_Head
        model = WDL_Head
    elif model_name.startswith('dcn'):
        from models import DCN_Head
        model = DCN_Head
    elif model_name.startswith('deepfm'):
        from models import DeepFM_Head
        model = DeepFM_Head
    else:
        raise NotImplementedError
    args.model = model
    worker(args)
