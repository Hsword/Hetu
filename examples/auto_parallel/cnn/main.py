from models import AlexNet, VGG19, ResNet101, InceptionV3, WideResNet50, WideResNet101
import hetu as ht
import numpy as np
import argparse
from time import time


# flags:
# pix (in profiler)
# overlap
# nccl collectives
# share embedding
# different model size
# different batch size
if __name__ == "__main__":
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        help='model to be used')
    parser.add_argument('--batch-size', type=int,
                        default=32, help='batch size')
    parser.add_argument('--lr', type=float,
                        default=0.001, help='learning rate')
    parser.add_argument('--l2reg', type=float,
                        default=0.0001, help='l2 regularizer')
    parser.add_argument('--log-iterations', type=int, default=10,
                        help='iterations executed before logging')
    parser.add_argument('--ignore-iter', type=int, default=1,
                        help='number of iterations that ignored')
    parser.add_argument('--log-num', type=int,
                        default=5, help='number of logs')
    parser.add_argument('--strategy', type=str, default='dp',
                        help='should be none, dp, mp, owt, flexflow, optcnn, gpipe, pipedream')
    parser.add_argument('--save-path', type=str, default='test.json',
                        help='saving path for searched strategies')
    parser.add_argument('--load-path', type=str, default=None,
                        help='loading path for searched strategies')
    parser.add_argument('--save-dir', type=str, default='temp',
                        help='saving directory for searched strategies')
    parser.add_argument('--load-dir', type=str, default=None,
                        help='loading directory for searched strategies')
    parser.add_argument('--batch-num-factor', type=int, default=1,
                        help='valid when using pipelines, number of micro batch')
    parser.add_argument('--nooverlap', action='store_true',
                        help='cancel overlap')
    parser.add_argument('--nopix', action='store_true',
                        help='cancel pix')
    parser.add_argument('--nonccl', action='store_true',
                        help='cancel nccl')
    args = parser.parse_args()

    args.strategy = args.strategy.lower()
    assert args.strategy in (
        'none', 'dp', 'mp', 'owt', 'flexflow', 'optcnn', 'gpipe', 'pipedream', 'pipeopt')
    is_pipeline = args.strategy in ('gpipe', 'pipedream', 'pipeopt')
    if is_pipeline:
        args.batch_size //= args.batch_num_factor
        assert args.batch_size > 0

    opt = ht.optim.SGDOptimizer(learning_rate=args.lr, l2reg=args.l2reg)
    if args.model == 'alexnet':
        model = AlexNet()
    elif args.model == 'vgg19':
        model = VGG19()
    elif args.model == 'resnet101':
        model = ResNet101()
    elif args.model == 'inceptionv3':
        model = InceptionV3()
    elif args.model == 'wresnet50':
        model = WideResNet50()
    elif args.model == 'wresnet101':
        model = WideResNet101()
    else:
        assert False
    crop_size = 299 if args.model == 'inceptionv3' else 224

    x = ht.placeholder_op(name='x')
    y_ = ht.placeholder_op(name='y_')

    if args.model != 'inceptionv3':
        y = model(x)
        loss = ht.softmaxcrossentropy_sparse_op(y, y_)
        loss = ht.reduce_mean_op(loss, [0])
    else:
        y, aux = model(x)
        loss = ht.softmaxcrossentropy_sparse_op(y, y_)
        loss_aux = ht.softmaxcrossentropy_sparse_op(aux, y_)
        loss = loss + 0.3 * loss_aux
        loss = ht.reduce_mean_op(loss, [0])

    train_op = opt.minimize(loss)

    eval_nodes = [loss, y, train_op]
    feed_shapes = {x: (args.batch_size, 3, crop_size,
                       crop_size), y_: (args.batch_size,)}
    print('The feed shapes are:', feed_shapes)
    if args.strategy == 'dp':
        strategy = ht.dist.DataParallel(aggregate='allreduce')
    elif args.strategy == 'mp':
        strategy = ht.dist.ModelParallel4CNN()
    elif args.strategy == 'owt':
        strategy = ht.dist.OneWeirdTrick4CNN(feed_shapes)
    elif args.strategy == 'flexflow':
        strategy = ht.dist.FlexFlowSearching(
            feed_shapes, unit_round_budget=1, save_path=args.save_path, load_path=args.load_path, load_with_simulate=True, pix=not args.nopix)
    elif args.strategy == 'optcnn':
        strategy = ht.dist.OptCNNSearching(
            feed_shapes, save_path=args.save_path, load_path=args.load_path, load_with_simulate=True, pix=not args.nopix)
    elif args.strategy == 'gpipe':
        strategy = ht.dist.GPipeSearching(
            feed_shapes, save_path=args.save_path, load_path=args.load_path, pix=not args.nopix)
    elif args.strategy == 'pipedream':
        strategy = ht.dist.PipeDreamSearching(
            feed_shapes, batch_num_factor=args.batch_num_factor, save_path=args.save_path, load_path=args.load_path, load_with_simulate=True, pix=not args.nopix)
    elif args.strategy == 'pipeopt':
        strategy = ht.dist.PipeOptSearching(feed_shapes, batch_num_factor=args.batch_num_factor,
                                            save_dir=args.save_dir, load_dir=args.load_dir, save_path=args.save_path, load_path=args.load_path, load_with_simulate=True, pix=not args.nopix)
    elif args.strategy == 'none':
        strategy = None
    else:
        assert False
    start = time()
    if is_pipeline:
        executor = ht.Executor(
            eval_nodes, dist_strategy=strategy, pipeline='gpipe', overlap=not args.nooverlap, use_nccl_collectives=not args.nonccl)
        if args.strategy == 'pipeopt':
            batch_num = strategy.batch_num
        else:
            batch_num = args.batch_num_factor * executor.config.nrank
        print('In pipeline strategy, the batch number is', batch_num)
    elif args.strategy == 'none':
        executor = ht.Executor(eval_nodes, ctx=ht.gpu(0),
                               overlap=not args.nooverlap, use_nccl_collectives=not args.nonccl)
    else:
        executor = ht.Executor(
            eval_nodes, dist_strategy=strategy, overlap=not args.nooverlap, use_nccl_collectives=not args.nonccl)
    ending = time()
    print('executor initiated time:', ending - start)

    # use synthetic data
    if args.strategy == 'pipeopt':
        args.batch_size = strategy.batch_size
    num_batches = max(
        batch_num+3, args.log_iterations) if is_pipeline else args.log_iterations
    image_shape = [args.batch_size, 3, crop_size, crop_size]
    if x.reshaped:
        for i in range(len(image_shape)):
            if i in x.parts:
                image_shape[i] //= x.parts[i]
        x.reshaped = False
    target_shape = [args.batch_size, ]
    if y_.reshaped:
        for i in range(len(target_shape)):
            if i in y_.parts:
                target_shape[i] //= y_.parts[i]
        y_.reshaped = False
    ctx = executor.config.context
    images = [ht.array(np.random.normal(loc=0.0, scale=0.1, size=image_shape).astype(
        np.float32), ctx=ctx) for _ in range(num_batches)]
    targets = [ht.array(np.random.randint(0, 1000, size=target_shape).astype(
        np.float32), ctx=ctx) for _ in range(num_batches)]

    # training
    running_time = 0
    for i in range(args.log_num):
        loss_all = 0
        acc_all = 0
        cnt = 0
        start = time()
        while cnt < args.log_iterations:
            image = images[cnt]
            target = targets[cnt]
            if is_pipeline:
                loss_val, predict_y, _ = executor.run(feed_dict=[{x: images[j], y_: targets[j]} for j in range(
                    batch_num)], convert_to_numpy_ret_vals=True, batch_num=batch_num)
            else:
                loss_val, predict_y, _ = executor.run(
                    feed_dict={x: image, y_: target}, convert_to_numpy_ret_vals=True)
            if loss_val is not None:
                loss_all += loss_val[0]
                # acc_all += np.sum(np.equal(target, np.argmax(predict_y, 1)))
            cnt += 1
        # loss_all, acc_all = list(executor.reduceMean([loss_all, acc_all]))
        if loss_all > 0:
            loss_all /= args.log_iterations
            # acc_all /= args.log_iterations
            print('Train loss = %f' % loss_all)
            # print('Train accuracy = %f' % acc_all)
        end = time()
        during_time = end - start
        if executor.rank in (0, None):
            print("Running time of current epoch = %fs" % (during_time))
        if i >= args.ignore_iter:
            running_time += during_time
    if executor.rank in (0, None):
        print("*"*50)
        all_iterations = (args.log_num - args.ignore_iter) * \
            args.log_iterations
        print("Running time of total %d iterations = %fs; each iteration time = %fms" %
              (all_iterations, running_time, running_time / all_iterations * 1000))
