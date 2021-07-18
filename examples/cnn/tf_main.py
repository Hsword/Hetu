import tensorflow as tf
import tf_models
import hetu as ht
import numpy as np
import argparse
from time import time
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def print_rank0(msg):
    logger.info(msg)


if __name__ == "__main__":
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        help='model to be tested')
    parser.add_argument('--dataset', type=str, required=True,
                        help='dataset to be trained on')
    parser.add_argument('--batch-size', type=int,
                        default=128, help='batch size')
    parser.add_argument('--learning-rate', type=float,
                        default=0.1, help='learning rate')
    parser.add_argument('--opt', type=str, default='sgd',
                        help='optimizer to be used, default sgd; sgd / momentum / adagrad / adam')
    parser.add_argument('--num-epochs', type=int,
                        default=20, help='epoch number')
    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu to be used, -1 means cpu')
    parser.add_argument('--validate', action='store_true',
                        help='whether to use validation')
    parser.add_argument('--timing', action='store_true',
                        help='whether to time the training phase')
    args = parser.parse_args()

    if args.gpu == -1:
        device = '/cpu:0'
        print_rank0('Use CPU.')
    else:
        device = '/gpu:%d' % args.gpu
        print_rank0('Use GPU %d.' % args.gpu)

    print_rank0("Training {} on TensorFlow".format(args.model))
    assert args.model in ['tf_cnn_3_layers', 'tf_lenet', 'tf_logreg', 'tf_lstm', 'tf_mlp', 'tf_resnet18', 'tf_resnet34', 'tf_rnn', 'tf_vgg16', 'tf_vgg19'], \
        'Model not supported now.'
    model = eval('tf_models.' + args.model)

    assert args.dataset in ['MNIST', 'CIFAR10', 'CIFAR100', 'ImageNet']
    dataset = args.dataset

    assert args.opt in ['sgd', 'momentum', 'nesterov',
                        'adagrad', 'adam'], 'Optimizer not supported!'
    if args.opt == 'sgd':
        print_rank0('Use SGD Optimizer.')
        opt = tf.train.GradientDescentOptimizer(
            learning_rate=args.learning_rate)
    elif args.opt == 'momentum':
        print_rank0('Use Momentum Optimizer.')
        opt = tf.train.MomentumOptimizer(
            learning_rate=args.learning_rate, momentum=0.9)
    elif args.opt == 'nesterov':
        print_rank0('Use Nesterov Momentum Optimizer.')
        opt = tf.train.MomentumOptimizer(
            learning_rate=args.learning_rate, momentum=0.9, use_nesterov=True)
    elif args.opt == 'adagrad':
        print_rank0('Use AdaGrad Optimizer.')
        opt = tf.train.AdagradOptimizer(learning_rate=args.learning_rate)
    else:
        print_rank0('Use Adam Optimizer.')
        opt = tf.train.AdamOptimizer(learning_rate=args.learning_rate)

    # model definition
    print_rank0('Building model...')
    with tf.device(device):
        if dataset == 'MNIST':
            x = tf.placeholder(dtype=tf.float32, shape=(None, 784), name='x')
            y_ = tf.placeholder(dtype=tf.float32, shape=(None, 10), name='y_')
            loss, y = model(x, y_)
        elif dataset == 'CIFAR10':
            if args.model == "tf_mlp":
                x = tf.placeholder(
                    dtype=tf.float32, shape=(None, 3072), name='x')
                y_ = tf.placeholder(
                    dtype=tf.float32, shape=(None, 10), name='y_')
            else:
                x = tf.placeholder(dtype=tf.float32, shape=(
                    None, 32, 32, 3), name='x')
                y_ = tf.placeholder(
                    dtype=tf.float32, shape=(None, 10), name='y_')
            loss, y = model(x, y_, 10)
        elif dataset == 'CIFAR100':
            x = tf.placeholder(dtype=tf.float32, shape=(
                None, 32, 32, 3), name='x')
            y_ = tf.placeholder(dtype=tf.float32, shape=(None, 100), name='y_')
            loss, y = model(x, y_, 100)

        train_op = opt.minimize(loss)

    # data loading
    print_rank0('Loading %s data...' % dataset)
    if dataset == 'MNIST':
        datasets = ht.data.mnist()
        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]
        n_train_batches = train_set_x.shape[0] // args.batch_size
        n_valid_batches = valid_set_x.shape[0] // args.batch_size
        # train_set_x: (50000, 784), train_set_y: (50000,)
        # valid_set_x: (10000, 784), valid_set_y: (10000,)
    elif dataset == 'CIFAR10':
        train_set_x, train_set_y, valid_set_x, valid_set_y = ht.data.tf_normalize_cifar(
            num_class=10)
        n_train_batches = train_set_x.shape[0] // args.batch_size
        n_valid_batches = valid_set_x.shape[0] // args.batch_size
        if args.model == "tf_mlp":
            train_set_x = train_set_x.reshape(train_set_x.shape[0], -1)
            valid_set_x = valid_set_x.reshape(valid_set_x.shape[0], -1)
        # train_set_x: (50000, 32, 32, 3), train_set_y: (50000,)
        # valid_set_x: (10000, 32, 32, 3), valid_set_y: (10000,)
    elif dataset == 'CIFAR100':
        train_set_x, train_set_y, valid_set_x, valid_set_y = ht.data.tf_normalize_cifar(
            num_class=100)
        n_train_batches = train_set_x.shape[0] // args.batch_size
        n_valid_batches = valid_set_x.shape[0] // args.batch_size
        # train_set_x: (50000, 32, 32, 3), train_set_y: (50000,)
        # valid_set_x: (10000, 32, 32, 3), valid_set_y: (10000,)
    else:
        raise NotImplementedError

    # training
    print_rank0("Start training loop...")
    running_time = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(args.num_epochs + 1):
            print_rank0("Epoch %d" % i)
            loss_all = 0
            batch_num = 0
            if args.timing:
                start = time()
            correct_predictions = []
            for minibatch_index in range(n_train_batches):
                minibatch_start = minibatch_index * args.batch_size
                minibatch_end = (minibatch_index + 1) * args.batch_size
                x_val = train_set_x[minibatch_start:minibatch_end]
                y_val = train_set_y[minibatch_start:minibatch_end]
                loss_val, predict_y, _ = sess.run([loss, y, train_op],
                                                  feed_dict={x: x_val, y_: y_val})
                correct_prediction = np.equal(
                    np.argmax(y_val, 1),
                    np.argmax(predict_y, 1)).astype(np.float32)
                correct_predictions.extend(correct_prediction)
                batch_num += 1
                loss_all += loss_val
            loss_all /= batch_num
            accuracy = np.mean(correct_predictions)
            print_rank0("Train loss = %f" % loss_all)
            print_rank0("Train accuracy = %f" % accuracy)

            if args.timing:
                end = time()
                print_rank0("Running time of current epoch = %fs" %
                            (end - start))
                if i != 0:
                    running_time += (end - start)

            if args.validate:
                val_loss_all = 0
                batch_num = 0
                correct_predictions = []
                for minibatch_index in range(n_valid_batches):
                    minibatch_start = minibatch_index * args.batch_size
                    minibatch_end = (minibatch_index + 1) * args.batch_size
                    valid_x_val = valid_set_x[minibatch_start:minibatch_end]
                    valid_y_val = valid_set_y[minibatch_start:minibatch_end]
                    loss_val, valid_y_predicted = sess.run([loss, y],
                                                           feed_dict={x: valid_x_val, y_: valid_y_val})
                    correct_prediction = np.equal(
                        np.argmax(valid_y_val, 1),
                        np.argmax(valid_y_predicted, 1)).astype(np.float32)
                    correct_predictions.extend(correct_prediction)
                    val_loss_all += loss_all
                    batch_num += 1
                val_loss_all /= batch_num
                accuracy = np.mean(correct_predictions)
                print_rank0("Validation loss = %f" % val_loss_all)
                print_rank0("Validation accuracy = %f" % accuracy)
        print_rank0("*"*50)
        print_rank0("Running time of total %d epoch = %fs" %
                    (args.num_epochs, running_time))
