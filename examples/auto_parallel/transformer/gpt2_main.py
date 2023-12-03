import hetu as ht
from models import HetuGPT2
from models import GPT2Config
import numpy as np
import time
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int,
                        default=8, help='batch size')
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
                        help='whether using gpt2 medium')
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

    lr = 1e-4

    if args.large:
        max_split = 16
        config = GPT2Config(
            vocab_size=50272,
            n_embd=1024,
            n_layer=24,
            n_inner=None,
            n_positions=1024,
            n_head=16,
            activation_function="relu",
            # resid_pdrop=0.,
            # embd_pdrop=0.,
            # attn_pdrop=0.,
            layer_norm_epsilon=1e-12,
            share_embedding=not is_pipeline,
        )
    else:
        max_split = 4
        config = GPT2Config(
            vocab_size=50272,
            n_embd=768,
            n_layer=12,
            n_inner=None,
            n_positions=1024,
            n_head=12,
            activation_function="relu",
            # resid_pdrop=0.,
            # embd_pdrop=0.,
            # attn_pdrop=0.,
            layer_norm_epsilon=1e-12,
            share_embedding=not is_pipeline,
        )
    seq_len = config.n_positions

    model = HetuGPT2(config=config)

    input_ids = ht.Variable(name='input_ids', trainable=False)
    attention_mask = ht.Variable(name='attention_mask', trainable=False)
    position_ids = ht.Variable(name='position_ids', trainable=False)

    loss, logits = model(
        input_ids, attention_mask, position_ids, input_ids)

    opt = ht.optim.AdamOptimizer(
        learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-8, l2reg=0.01)
    train_op = opt.minimize(loss)

    feed_shapes = {
        input_ids: (args.batch_size * seq_len,),
        attention_mask: (args.batch_size, seq_len),
        position_ids: (args.batch_size * seq_len,),
    }
    print('The feed shapes are:', feed_shapes)
    if args.strategy == 'dp':
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
            feed_shapes, batch_num_factor=args.batch_num_factor, batch_size=args.batch_size, save_dir=args.save_dir, load_dir=args.load_dir, save_path=args.save_path, load_path=args.load_path, pix=not args.nopix)
    else:
        assert args.strategy == 'none'
    start = time.time()
    if is_pipeline:
        executor = ht.Executor([loss, train_op], seed=123,
                               dist_strategy=strategy, pipeline='gpipe', enable_lazy=True, overlap=not args.nooverlap, use_nccl_collectives=not args.nonccl)
        if args.strategy == 'pipeopt':
            batch_num = strategy.batch_num
        else:
            batch_num = args.batch_num_factor * executor.config.nrank
        print('In pipeline strategy, the batch number is', batch_num)
    elif args.strategy == 'none':
        executor = ht.Executor(
            [loss, train_op], seed=123, ctx=ht.gpu(0), enable_lazy=True, overlap=not args.nooverlap, use_nccl_collectives=not args.nonccl)
    else:
        executor = ht.Executor(
            [loss, train_op], seed=123, dist_strategy=strategy, enable_lazy=True, overlap=not args.nooverlap, use_nccl_collectives=not args.nonccl)
    ending = time.time()
    print('executor initiated time:', ending - start)

    # use synthetic data
    num_batches = max(
        batch_num+3, 10) if is_pipeline else 10
    input_id_vals = []
    attention_masks = []
    ctx = executor.config.context
    batch_size = args.batch_size
    if args.strategy == 'pipeopt':
        batch_size = strategy.batch_size
    for i in range(num_batches):
        # (batch_size, max_position)
        length = np.random.randint(400, seq_len, size=batch_size)
        input_ids_val = np.random.randint(
            config.vocab_size, size=(batch_size, seq_len)).reshape((-1)).astype(np.float32)
        attention_mask_val = np.array([[x <= length[i] for x in range(
            seq_len)] for i in range(batch_size)], dtype=int).astype(np.float32)
        if input_ids.reshaped:
            input_ids_val = input_ids.reshape_tensor(input_ids_val)
        if attention_mask.reshaped:
            attention_mask_val = attention_mask.reshape_tensor(
                attention_mask_val)
        input_id_vals.append(ht.array(input_ids_val, ctx=ctx))
        attention_masks.append(ht.array(attention_mask_val, ctx=ctx))
    input_ids.reshaped = False
    attention_mask.reshaped = False
    position_ids_arr = np.tile(np.arange(config.n_positions), batch_size)
    if position_ids.reshaped:
        position_ids_arr = position_ids.reshape_tensor(position_ids_arr)
    position_ids.reshaped = False
    position_ids_arr = ht.array(position_ids_arr, ctx=ctx)

    def get_batch():
        nonlocal cnt
        # (batch_size, max_position)
        feed_dict = {
            input_ids: input_id_vals[cnt],
            attention_mask: attention_masks[cnt],
            position_ids: position_ids_arr,
        }
        cnt = (cnt + 1) % num_batches
        return feed_dict

    cnt = 0
    for i in range(args.log_num):
        if i == args.ignore_iter:
            start = time.time()
        start_time = time.time()

        if is_pipeline:
            loss_val, _ = executor.run(
                feed_dict=[get_batch() for j in range(batch_num)], batch_num=batch_num)
            if loss_val is not None:
                loss_val = [l.asnumpy() for l in loss_val]
        else:
            loss_val, _ = executor.run(feed_dict=get_batch())
            if loss_val is not None:
                loss_val = loss_val.asnumpy()
        end_time = time.time()
        if loss_val is not None:
            print('(Iteration {}): Loss = {}, Time ={}'.format(
                i, loss_val, end_time-start_time))

    end = time.time()
    running_time = (end - start)
    all_iterations = (args.log_num - args.ignore_iter)
    if executor.rank in (0, None):
        print("Running time of total %d iterations = %fs; each iteration time = %fms" %
              (all_iterations, running_time, running_time / all_iterations * 1000))


if __name__ == '__main__':
    main()
