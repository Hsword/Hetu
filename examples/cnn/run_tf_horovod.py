import os
import numpy as np
import tensorflow as tf
import tf_models
import time
import argparse
from tqdm import tqdm
from sklearn import metrics
import horovod.tensorflow as hvd
import hetu as ht
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def print_rank0(msg):
    if rank % 8 == 0:
        logger.info(msg)


def pop_env():
    for k in ['https_proxy', 'http_proxy']:
        if k in os.environ:
            os.environ.pop(k)


pop_env()

# horovodrun -np 8 -H localhost:8 python run_tf_horovod.py --model
# horovodrun -np 8 --start-timeout 300 -H node1:4,node2:4 python run_tf_horovod.py --model
# horovodrun -np 16 --start-timeout 3000 -H node1:8,node2:8
#    python /home/public/nxn/Athena-master/examples/cnn/run_tf_horovod.py --model tf_rnn


# if using multi nodes setting in conda, need to modify /etc/bash.bashrc
# we can also use mpirun (default gloo):
# ../build/_deps/openmpi-build/bin/mpirun -mca btl_tcp_if_include enp97s0f0 --bind-to none --map-by slot\
#  -x NCCL_SOCKET_IFNAME=enp97s0f0 -H node2:8,node3:8 --allow-run-as-root python run_tf_horovod.py --model
'''
def train(model, args):
    hvd.init()

    def get_current_shard(data):
        part_size = data.shape[0] // hvd.size()
        start = part_size * hvd.rank()
        end = start + part_size if hvd.rank() != hvd.size() - 1 else data.shape[0]
        return data[start:end]

    batch_size = 128
    if args.model == 'tf_resnet34':
        train_images, train_labels, test_images,\
                test_labels = ht.data.tf_normalize_cifar10()
        x = tf.compat.v1.placeholder(tf.float32, [batch_size, 32, 32, 3])
        y_ = y_ = tf.compat.v1.placeholder(tf.float32, [batch_size, 10])
    else:
        datasets = ht.data.mnist()
        train_images, train_labels = datasets[0]
        test_images, test_labels = datasets[2]
        x = tf.compat.v1.placeholder(tf.float32, [batch_size, 784])
        y_ = y_ = tf.compat.v1.placeholder(tf.float32, [batch_size, 10])


    n_train_batches = train_images.shape[0] // batch_size

    loss, y = model(x, y_)
    opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)

    global_step = tf.train.get_or_create_global_step()
    # here in DistributedOptimizer by default all tensor are reduced on GPU
    # can use device_sparse=xxx, device_dense=xxx to modify
    # if using device_sparse='/cpu:0', the performance degrades
    train_op = hvd.DistributedOptimizer(opt).minimize(loss, global_step=global_step)

    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True, visible_device_list=str(hvd.local_rank()))
    # here horovod default use gpu to initialize, which will cause OOM
    hooks = [hvd.BroadcastGlobalVariablesHook(0, device='/cpu:0')]
    sess = tf.compat.v1.train.MonitoredTrainingSession(hooks=hooks, config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

    iterations = train_images.shape[0] // batch_size
    total_epoch = 10
    start_index = 0
    total_time = 0
    for ep in range(total_epoch + 1):
        print("epoch %d" % ep)
        st_time = time.time()
        train_loss, train_acc = [], []
        for it in range(n_train_batches):
            x_val = train_images[start_index: start_index + batch_size]
            y_val = train_labels[start_index : start_index+batch_size]
            start_index += batch_size
            if start_index + batch_size > train_images.shape[0]:
                start_index = 0
            loss_val = sess.run([loss, y, y_, train_op], feed_dict={x:x_val, y_:y_val})
            pred_val = loss_val[1]
            true_val = loss_val[2]
            acc_val = np.equal(
                true_val,
                pred_val > 0.5)
            train_loss.append(loss_val[0])
            train_acc.append(acc_val)
        tra_accuracy = np.mean(train_acc)
        tra_loss = np.mean(train_loss)
        en_time = time.time()
        train_time = en_time - st_time
        if ep != 0:
            total_time += train_time
        printstr = "train_loss: %.4f, train_acc: %.4f, train_time: %.4f"\
                    % (tra_loss, tra_accuracy, train_time)

    print("training time:", total_time)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="model to be tested")
    parser.add_argument("--all", action="store_true", help="whether to use all data")
    args = parser.parse_args()
    raw_model = args.model
    import tf_models
    model = eval('tf_models.' + raw_model)
    print('Model:', raw_model)
    train(model, args)

if __name__ == '__main__':
    main()
'''

if __name__ == "__main__":

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
    parser.add_argument('--validate', action='store_true',
                        help='whether to use validation')
    parser.add_argument('--timing', action='store_true',
                        help='whether to time the training phase')
    args = parser.parse_args()

    hvd.init()
    global rank
    rank = hvd.rank()
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

    if dataset == 'MNIST':
        x = tf.compat.v1.placeholder(
            dtype=tf.float32, shape=(None, 784), name='x')
        y_ = tf.compat.v1.placeholder(
            dtype=tf.float32, shape=(None, 10), name='y_')
        loss, y = model(x, y_)
    elif dataset == 'CIFAR10':
        if args.model == "tf_mlp":
            x = tf.compat.v1.placeholder(
                dtype=tf.float32, shape=(None, 3072), name='x')
            y_ = tf.compat.v1.placeholder(
                dtype=tf.float32, shape=(None, 10), name='y_')
        else:
            x = tf.compat.v1.placeholder(
                dtype=tf.float32, shape=(None, 32, 32, 3), name='x')
            y_ = tf.compat.v1.placeholder(
                dtype=tf.float32, shape=(None, 10), name='y_')
        loss, y = model(x, y_, 10)
    elif dataset == 'CIFAR100':
        x = tf.compat.v1.placeholder(
            dtype=tf.float32, shape=(None, 32, 32, 3), name='x')
        y_ = tf.compat.v1.placeholder(
            dtype=tf.float32, shape=(None, 100), name='y_')
        loss, y = model(x, y_, 100)

    global_step = tf.train.get_or_create_global_step()
    # here in DistributedOptimizer by default all tensor are reduced on GPU
    # can use device_sparse=xxx, device_dense=xxx to modify
    # if using device_sparse='/cpu:0', the performance degrades
    train_op = hvd.DistributedOptimizer(
        opt).minimize(loss, global_step=global_step)

    gpu_options = tf.compat.v1.GPUOptions(
        allow_growth=True, visible_device_list=str(hvd.local_rank()))
    # here horovod default use gpu to initialize, which will cause OOM
    hooks = [hvd.BroadcastGlobalVariablesHook(0, device='/cpu:0')]
    sess = tf.compat.v1.train.MonitoredTrainingSession(
        hooks=hooks, config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

    # sess.run(tf.compat.v1.global_variables_initializer())

    # training
    print_rank0("Start training loop...")
    running_time = 0
    for i in range(args.num_epochs + 1):
        print_rank0("Epoch %d" % i)
        loss_all = 0
        batch_num = 0
        if args.timing:
            start = time.time()
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
            end = time.time()
            print_rank0("Running time of current epoch = %fs" % (end - start))
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
