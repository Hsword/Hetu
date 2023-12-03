from models import AlexNet
import hetu as ht
import os
import numpy as np
import argparse


if __name__ == "__main__":
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float,
                        default=0.1, help='learning rate')
    parser.add_argument('--l2reg', type=float,
                        default=0., help='l2 regularizer')
    parser.add_argument('--strategy', type=str, default=None,
                        help='should be None, dp, mp, owt')
    parser.add_argument('--load-path', type=str, default=None,
                        help='loading path for searched strategies')
    args = parser.parse_args()

    if args.strategy is not None:
        args.strategy = args.strategy.lower()
    if args.strategy == 'none':
        args.strategy = None
    assert args.strategy in (None, 'dp', 'mp', 'owt')

    single_execution = args.strategy is None and args.load_path is None

    if args.strategy is not None and args.load_path is None:
        # TODO: divide the oneslike op in loss if we have multiple losses!
        # now for workaround, we only test 2 workers case
        # and ignore l2reg
        args.lr /= 2
    opt = ht.optim.SGDOptimizer(learning_rate=args.lr, l2reg=args.l2reg)
    model = AlexNet(dropout=0., bias=False)
    crop_size = 224
    global_batch_size = 128
    log_num = 3
    work_dir = 'test_strategy'
    if single_execution:
        os.makedirs(work_dir, exist_ok=True)

    def get_file(path):
        return os.path.join(work_dir, path)

    # use synthetic data
    if single_execution:
        images = [np.random.normal(loc=0.0, scale=1, size=(
            global_batch_size, 3, crop_size, crop_size)).astype(np.float32) for _ in range(log_num)]
        targets = [np.random.randint(0, 1000, size=(
            global_batch_size,)).astype(np.float32) for _ in range(log_num)]
        np.save(get_file('images.npy'), images)
        np.save(get_file('targets.npy'), targets)
    else:
        images = np.load(get_file('images.npy'))
        targets = np.load(get_file('targets.npy'))

    x = ht.placeholder_op(name='x')
    y_ = ht.placeholder_op(name='y_')

    y = model(x)
    loss = ht.softmaxcrossentropy_sparse_op(y, y_)
    loss = ht.reduce_mean_op(loss, [0])
    train_op = opt.minimize(loss)
    eval_nodes = [loss, y, train_op]

    if single_execution:
        strategy = None
    elif args.load_path is not None:
        strategy = ht.dist.BaseSearchingStrategy({x: (global_batch_size, 3, crop_size, crop_size), y_: (
            global_batch_size,)}, load_path=args.load_path, include_duplicate=False)
    elif args.strategy == 'dp':
        strategy = ht.dist.DataParallel(aggregate='allreduce')
    elif args.strategy == 'mp':
        strategy = ht.dist.ModelParallel4CNN()
    elif args.strategy == 'owt':
        strategy = ht.dist.OneWeirdTrick4CNN()

    if single_execution:
        executor = ht.Executor(eval_nodes, ctx=ht.gpu(0))
        executor.save(work_dir, 'alexnet.bin')
    else:
        executor = ht.Executor(eval_nodes, dist_strategy=strategy)
        executor.load(work_dir, 'alexnet.bin', consider_splits=True)

    for i in range(log_num):
        image = images[i]
        target = targets[i]
        if single_execution:
            loss_val, predict_y, _ = executor.run(
                feed_dict={x: image, y_: target}, convert_to_numpy_ret_vals=True)
            np.save(get_file('gt_loss{}.npy'.format(i)), loss_val)
            np.save(get_file('gt_predict_y{}.npy'.format(i)), predict_y)
        else:
            if args.strategy == 'dp':
                local_batch_size = global_batch_size // executor.config.nrank
                cur_slice = slice(
                    executor.rank * local_batch_size, (executor.rank + 1) * local_batch_size)
                image = image[cur_slice]
                target = target[cur_slice]
            loss_val, predict_y, _ = executor.run(
                feed_dict={x: image, y_: target}, convert_to_numpy_ret_vals=True)
            if args.load_path is None:
                if args.strategy == 'dp':
                    test_loss_val = executor.reduceMean(loss_val)
                    test_predict_y = executor.gatherPredict(predict_y)
                elif args.strategy in ('mp', 'owt'):
                    test_loss_val = executor.reduceMean(loss_val)
                    test_predict_y = executor.gatherPredict(predict_y)
                    split_test_predict_y = np.split(
                        test_predict_y, executor.config.nrank, axis=0)
                    test_predict_y = np.concatenate(
                        split_test_predict_y, axis=1)
            else:
                test_predict_y = predict_y
                test_loss_val = loss_val
            if executor.rank == 0:
                gt_loss_val = np.load(get_file('gt_loss{}.npy'.format(i)))
                gt_predict_y = np.load(
                    get_file('gt_predict_y{}.npy'.format(i)))
                np.testing.assert_allclose(
                    gt_predict_y, test_predict_y, atol=1e-3)
                np.testing.assert_allclose(
                    gt_loss_val, test_loss_val, rtol=1e-5)
                print('Pass test with {}, {}'.format(gt_loss_val, loss_val))
