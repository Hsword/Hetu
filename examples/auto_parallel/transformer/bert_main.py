import hetu as ht
from models import HetuBert
from models import BertConfig
from load_data import DataLoader
import numpy as np
import time
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int,
                        default=18, help='batch size')
    parser.add_argument('--strategy', type=str, default='flexflow',
                        help='should be none, dp, mp, megatronlm, flexflow, optcnn, gpipe, pipedream, pipeopt')
    parser.add_argument('--save-path', type=str, default='test.json',
                        help='saving path for searched strategies')
    parser.add_argument('--load-path', type=str, default=None,
                        help='loading path for searched strategies')
    parser.add_argument('--save-dir', type=str, default='temp',
                        help='saving directory for searched strategies')
    parser.add_argument('--load-dir', type=str, default=None,
                        help='loading directory for searched strategies')
    parser.add_argument('--ignore-iter', type=int, default=1,
                        help='number of iterations that ignored')
    parser.add_argument('--log-num', type=int,
                        default=5, help='number of logs')
    parser.add_argument('--batch-num-factor', type=int, default=1,
                        help='valid when using pipelines, number of micro batch')
    parser.add_argument('--nooverlap', action='store_true',
                        help='cancel overlap')
    parser.add_argument('--large', action='store_true',
                        help='whether using bert large')
    parser.add_argument('--nopix', action='store_true',
                        help='cancel pix')
    parser.add_argument('--nonccl', action='store_true',
                        help='cancel nccl')
    args = parser.parse_args()
    assert args.strategy in ('none', 'dp', 'mp', 'megatronlm', 'flexflow',
                             'optcnn', 'gpipe', 'pipedream', 'pipeopt')
    is_pipeline = args.strategy in ('gpipe', 'pipedream', 'pipeopt')
    if is_pipeline:
        args.batch_size //= args.batch_num_factor
        assert args.batch_size > 0

    num_epochs = 1
    lr = 1e-4

    if args.large:
        max_split = 16
        config = BertConfig(vocab_size=30528,
                            hidden_size=1024,
                            num_hidden_layers=24,
                            num_attention_heads=16,
                            intermediate_size=4096,
                            max_position_embeddings=512,
                            # attention_probs_dropout_prob=0.0,
                            # hidden_dropout_prob=0.0,
                            share_embedding=not is_pipeline)
    else:
        max_split = 4
        config = BertConfig(vocab_size=30528,
                            hidden_size=768,
                            num_hidden_layers=12,
                            num_attention_heads=12,
                            intermediate_size=3072,
                            max_position_embeddings=512,
                            # attention_probs_dropout_prob=0.0,
                            # hidden_dropout_prob=0.0,
                            share_embedding=not is_pipeline)

    model = HetuBert(config=config)

    input_ids = ht.Variable(name='input_ids', trainable=False)
    token_type_ids = ht.Variable(name='token_type_ids', trainable=False)
    position_ids = ht.Variable(name='position_ids', trainable=False)
    attention_mask = ht.Variable(name='attention_mask', trainable=False)

    masked_lm_labels = ht.Variable(name='masked_lm_labels', trainable=False)
    next_sentence_label = ht.Variable(
        name='next_sentence_label', trainable=False)

    loss_position_sum = ht.Variable(name='loss_position_sum', trainable=False)

    _, _, masked_lm_loss, next_sentence_loss = model(
        input_ids, attention_mask, token_type_ids, position_ids, masked_lm_labels, next_sentence_label)

    masked_lm_loss_mean = ht.div_op(ht.reduce_sum_op(
        masked_lm_loss, [0, ]), loss_position_sum)
    next_sentence_loss_mean = ht.reduce_mean_op(next_sentence_loss, [0])

    loss = masked_lm_loss_mean + next_sentence_loss_mean
    opt = ht.optim.AdamOptimizer(
        learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-8, l2reg=0.01)
    # opt = ht.optim.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-8)
    # opt = ht.optim.SGDOptimizer(learning_rate=lr)
    train_op = opt.minimize(loss)

    length = args.batch_size * config.max_position_embeddings
    feed_shapes = {
        input_ids: (length,),
        token_type_ids: (length,),
        position_ids: (length,),
        attention_mask: (args.batch_size, config.max_position_embeddings),
        masked_lm_labels: (length,),
        next_sentence_label: (args.batch_size,),
        loss_position_sum: (1,),
    }
    print('The feed shapes are:', feed_shapes)
    if args.strategy == 'none':
        strategy = None
    elif args.strategy == 'dp':
        strategy = ht.dist.DataParallel(aggregate='allreduce')
    elif args.strategy == 'mp':
        strategy = ht.dist.ModelParallel4LM()
    elif args.strategy == 'megatronlm':
        strategy = ht.dist.MegatronLM(max_split, save_path=args.save_path)
    elif args.strategy == 'flexflow':
        strategy = ht.dist.FlexFlowSearching(
            feed_shapes, batch_size=args.batch_size, unit_round_budget=1, save_path=args.save_path, load_path=args.load_path, pix=not args.nopix)
    elif args.strategy == 'optcnn':
        strategy = ht.dist.OptCNNSearching(
            feed_shapes, batch_size=args.batch_size, save_path=args.save_path, load_path=args.load_path, pix=not args.nopix, load_with_simulate=True)
    elif args.strategy == 'gpipe':
        strategy = ht.dist.GPipeSearching(
            feed_shapes, batch_size=args.batch_size, save_path=args.save_path, load_path=args.load_path, pix=not args.nopix)
    elif args.strategy == 'pipedream':
        strategy = ht.dist.PipeDreamSearching(
            feed_shapes, batch_num_factor=args.batch_num_factor, batch_size=args.batch_size, save_path=args.save_path, load_path=args.load_path, pix=not args.nopix, load_with_simulate=True)
    elif args.strategy == 'pipeopt':
        strategy = ht.dist.PipeOptSearching(
            feed_shapes, batch_num_factor=args.batch_num_factor, batch_size=args.batch_size, save_dir=args.save_dir, load_dir=args.load_dir, save_path=args.save_path, load_path=args.load_path, pix=not args.nopix, ignore_batch_size=[loss_position_sum])
    else:
        assert False
    start = time.time()
    if is_pipeline:
        executor = ht.Executor([masked_lm_loss_mean, next_sentence_loss_mean,
                                loss, train_op], seed=123, dist_strategy=strategy, pipeline='gpipe', enable_lazy=True, overlap=not args.nooverlap, use_nccl_collectives=not args.nonccl)
        if args.strategy == 'pipeopt':
            batch_num = strategy.batch_num
        else:
            batch_num = args.batch_num_factor * executor.config.nrank
        print('In pipeline strategy, the batch number is', batch_num)
    elif args.strategy == 'none':
        executor = ht.Executor([masked_lm_loss_mean, next_sentence_loss_mean,
                                loss, train_op], seed=123, ctx=ht.gpu(0), enable_lazy=True, overlap=not args.nooverlap, use_nccl_collectives=not args.nonccl)
    else:
        executor = ht.Executor([masked_lm_loss_mean, next_sentence_loss_mean,
                                loss, train_op], seed=123, dist_strategy=strategy, enable_lazy=True, overlap=not args.nooverlap, use_nccl_collectives=not args.nonccl)
    ending = time.time()
    print('executor initiated time:', ending - start)

    # use synthetic data
    batch_size = args.batch_size
    if args.strategy == 'pipeopt':
        batch_size = strategy.batch_size
    dataloader = DataLoader(dataset='bookcorpus', doc_num=200,
                            save_gap=200, batch_size=batch_size)
    dataloader.make_epoch_data()
    num_batches = max(
        batch_num+3, 10) if is_pipeline else 10
    var_str = ['input_ids', 'token_type_ids', 'attention_mask',
               'masked_lm_labels', 'next_sentence_label', 'loss_position_sum']
    str_to_var = {
        'input_ids': input_ids,
        'token_type_ids': token_type_ids,
        'attention_mask': attention_mask,
        'masked_lm_labels': masked_lm_labels,
        'next_sentence_label': next_sentence_label,
        'loss_position_sum': loss_position_sum,
    }
    ctx = executor.config.context
    values = {str_to_var[var]: [] for var in var_str}
    cur_pos_val = np.tile(
        np.arange(config.max_position_embeddings), batch_size)
    if position_ids.reshaped:
        cur_pos_val = position_ids.reshape_tensor(cur_pos_val)
    position_ids.reshaped = False
    values[position_ids] = ht.array(cur_pos_val, ctx=ctx)
    for i in range(num_batches):
        batch_data = dataloader.get_batch(i)
        for var in var_str:
            if var == 'loss_position_sum':
                cur_value = np.array(
                    [np.where(cur_value.reshape(-1) != -1)[0].shape[0]])
            else:
                cur_value = batch_data[var]
                if var != 'next_sentence_label':
                    cur_value = cur_value.reshape((-1,))
            cur_var = str_to_var[var]
            if cur_var.reshaped:
                cur_value = cur_var.reshape_tensor(cur_value)
            values[cur_var].append(ht.array(cur_value, ctx=ctx))
    for var in var_str:
        cur_var = str_to_var[var]
        cur_var.reshaped = False

    def get_batch():
        nonlocal cnt
        feed_dict = {str_to_var[var]: values[str_to_var[var]][cnt]
                     for var in var_str}
        cnt = (cnt + 1) % num_batches
        feed_dict[position_ids] = values[position_ids]
        return feed_dict

    cnt = 0
    for ep in range(num_epochs):
        for i in range(dataloader.batch_num):
            if i == args.ignore_iter:
                start = time.time()
            if i == args.log_num:
                end = time.time()
                running_time = (end - start)
                all_iterations = (args.log_num - args.ignore_iter)
                if executor.rank in (0, None):
                    print("Running time of total %d iterations = %fs; each iteration time = %fms" %
                          (all_iterations, running_time, running_time / all_iterations * 1000))
                exit()
            start_time = time.time()

            if is_pipeline:
                results = executor.run(
                    feed_dict=[get_batch() for j in range(batch_num)], batch_num=batch_num)
                masked_lm_loss_mean_out = np.mean(
                    [x.asnumpy() for x in results[0]]) if results[0] else None
                next_sentence_loss_mean_out = np.mean(
                    [x.asnumpy() for x in results[1]]) if results[1] else None
                loss_out = np.mean([x.asnumpy()
                                    for x in results[2]]) if results[2] else None
            else:
                feed_dict = get_batch()
                results = executor.run(feed_dict=feed_dict)
                masked_lm_loss_mean_out = results[0].asnumpy(
                ) if results[0] else None
                next_sentence_loss_mean_out = results[1].asnumpy(
                ) if results[1] else None
                loss_out = results[2].asnumpy() if results[2] else None
            end_time = time.time()
            if masked_lm_loss_mean_out is not None or next_sentence_loss_mean_out is not None or loss_out is not None:
                print('[Epoch {}] (Iteration {}): Loss = {}, MLM_loss = {}, NSP_loss = {}, Time ={}'.format(
                    ep, i, loss_out, masked_lm_loss_mean_out, next_sentence_loss_mean_out, end_time-start_time))


if __name__ == '__main__':
    main()
