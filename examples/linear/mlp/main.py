import hetu as ht
from hetu_mlp import mlp
import numpy as np
import argparse
import logging
from time import time
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def print_rank0(msg):
    if device_id == 0:
        logger.info(msg)


if __name__ == "__main__":
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int,
                        default=32, help='batch size')
    parser.add_argument('--learning-rate', type=float,
                        default=0.1, help='learning rate')
    parser.add_argument('--opt', type=str, default='sgd',
                        help='optimizer to be used, default sgd; sgd / momentum / adagrad / adam')
    parser.add_argument('--num-epochs', type=int,
                        default=10, help='epoch number')
    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu to be used, -1 means cpu')
    parser.add_argument('--validate', action='store_true',
                        help='whether to use validation')
    parser.add_argument('--timing', action='store_true',
                        help='whether to time the training phase')
    parser.add_argument('--comm-mode', default=None, help='communication mode')
    args = parser.parse_args()

    if args.comm_mode is not None:
        args.comm_mode = args.comm_mode.lower()
    assert args.comm_mode in (None, 'allreduce', 'ps', 'hybrid')

    args.opt = args.opt.lower()
    assert args.opt in ['sgd', 'momentum', 'nesterov',
                        'adagrad', 'adam'], 'Optimizer not supported!'

    if args.opt == 'sgd':
        opt = ht.optim.SGDOptimizer(learning_rate=args.learning_rate)
    elif args.opt == 'momentum':
        opt = ht.optim.MomentumOptimizer(learning_rate=args.learning_rate)
    elif args.opt == 'nesterov':
        opt = ht.optim.MomentumOptimizer(
            learning_rate=args.learning_rate, nesterov=True)
    elif args.opt == 'adagrad':
        opt = ht.optim.AdaGradOptimizer(
            learning_rate=args.learning_rate, initial_accumulator_value=0.1)
    else:
        opt = ht.optim.AdamOptimizer(learning_rate=args.learning_rate)

    train_set_x, train_set_y, valid_set_x, valid_set_y = ht.data.normalize_cifar(num_class=10)
    train_set_x = train_set_x.reshape(train_set_x.shape[0], -1)
    valid_set_x = valid_set_x.reshape(valid_set_x.shape[0], -1)

    # model definition
    x = ht.dataloader_op([
        ht.Dataloader(train_set_x, args.batch_size, 'train'),
        ht.Dataloader(valid_set_x, args.batch_size, 'validate'),
    ])
    y_ = ht.dataloader_op([
        ht.Dataloader(train_set_y, args.batch_size, 'train'),
        ht.Dataloader(valid_set_y, args.batch_size, 'validate'),
    ])

    loss, y = mlp(x, y_)

    train_op = opt.minimize(loss)

    eval_nodes = {'train': [loss, y, y_, train_op], 'validate': [loss, y, y_]}
    if args.comm_mode is None:
        if args.gpu < 0:
            ctx = ht.cpu()
        else:
            ctx = ht.gpu(args.gpu)
        executor = ht.Executor(eval_nodes, ctx=ctx)
    else:
        strategy = ht.dist.DataParallel(args.comm_mode)
        executor = ht.Executor(eval_nodes, dist_strategy=strategy)
    n_train_batches = executor.get_batch_num('train')
    n_valid_batches = executor.get_batch_num('validate')

    global device_id
    device_id = executor.rank
    if device_id is None:
        device_id = 0
    print_rank0("Training MLP on HETU")
    print_rank0('Use {} Optimizer.'.format(args.opt))
    print_rank0('Use data CIFAR10.')

    # training
    print_rank0("Start training loop...")
    running_time = 0
    for i in range(args.num_epochs + 1):
        print_rank0("Epoch %d" % i)
        loss_all = 0
        batch_num = 0
        if args.timing:
            start = time()
        correct_predictions = []
        for minibatch_index in range(n_train_batches):
            loss_val, predict_y, y_val, _ = executor.run(
                'train', eval_node_list=[loss, y, y_, train_op])
            # Loss for this minibatch
            predict_y = predict_y.asnumpy()
            y_val = y_val.asnumpy()
            loss_all += loss_val.asnumpy()
            batch_num += 1
            # Predict accuracy for this minibatch
            correct_prediction = np.equal(
                np.argmax(y_val, 1),
                np.argmax(predict_y, 1)).astype(np.float32)
            correct_predictions.extend(correct_prediction)

        loss_all /= batch_num
        accuracy = np.mean(correct_predictions)
        print_rank0("Train loss = %f" % loss_all)
        print_rank0("Train accuracy = %f" % accuracy)

        if args.timing:
            end = time()
            during_time = end - start
            print_rank0("Running time of current epoch = %fs" % (during_time))
            if i != 0:
                running_time += during_time
        if args.validate:
            val_loss_all = 0
            batch_num = 0
            correct_predictions = []
            for minibatch_index in range(n_valid_batches):
                loss_val, valid_y_predicted, y_val = executor.run(
                    'validate', eval_node_list=[loss, y, y_], convert_to_numpy_ret_vals=True)
                val_loss_all += loss_val
                batch_num += 1
                correct_prediction = np.equal(
                    np.argmax(y_val, 1),
                    np.argmax(valid_y_predicted, 1)).astype(np.float32)
                correct_predictions.extend(correct_prediction)

            val_loss_all /= batch_num
            accuracy = np.mean(correct_predictions)
            print_rank0("Validation loss = %f" % val_loss_all)
            print_rank0("Validation accuracy = %f" % accuracy)
    print_rank0("*"*50)
    print_rank0("Running time of total %d epoch = %fs" %
                (args.num_epochs, running_time))
