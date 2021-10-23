import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from pytorch_models import *
import hetu as ht
import numpy as np
import argparse
from time import time
import os
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def print_rank0(msg):
    if local_rank % 8 == 0:
        logger.info(msg)


def train(epoch=-1, net=None, data=None, label=None, batch_size=-1, criterion=None, optimizer=None):
    print_rank0('Epoch: %d' % epoch)
    n_train_batches = data.shape[0] // batch_size

    net.train()

    train_loss = 0
    correct = 0
    total = 0

    for minibatch_index in range(n_train_batches):
        minibatch_start = minibatch_index * args.batch_size
        minibatch_end = (minibatch_index + 1) * args.batch_size
        inputs = torch.Tensor(data[minibatch_start:minibatch_end])
        targets = torch.Tensor(label[minibatch_start:minibatch_end]).long()

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print_rank0("Train loss = %f" % (train_loss/(minibatch_index+1)))
    print_rank0("Train accuracy = %f" % (100.*correct/total))


def test(epoch=-1, net=None, data=None, label=None, batch_size=-1, criterion=None):
    net.eval()
    n_test_batches = data.shape[0] // batch_size
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for minibatch_index in range(n_test_batches):
            minibatch_start = minibatch_index * args.batch_size
            minibatch_end = (minibatch_index + 1) * args.batch_size
            inputs = torch.Tensor(data[minibatch_start:minibatch_end])
            targets = torch.Tensor(label[minibatch_start:minibatch_end]).long()

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print_rank0("Validation loss = %f" % (test_loss/(minibatch_index+1)))
        print_rank0("Validation accuracy = %f" % (100.*correct/total))


if __name__ == "__main__":
    # argument parser
    global local_rank
    local_rank = 0
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
    parser.add_argument('--distributed', action='store_true',
                        help='whether to distributed training')
    parser.add_argument('--local_rank', type=int, default=-1)
    args = parser.parse_args()

    if args.distributed == True:
        init_method = 'tcp://'
        master_ip = os.getenv('MASTER_ADDR', 'localhost')
        master_port = os.getenv('MASTER_PORT', '6000')
        init_method += master_ip + ':' + master_port
        rank = int(os.getenv('RANK', '0'))
        world_size = int(os.getenv("WORLD_SIZE", '1'))
        print("***"*50)
        print(init_method)
        torch.distributed.init_process_group(backend="nccl",
                                             world_size=world_size,
                                             rank=rank,
                                             init_method=init_method)

    if args.gpu == -1:
        device = 'cpu'
    else:
        if args.distributed == True:
            local_rank = rank % torch.cuda.device_count()
            torch.cuda.set_device(local_rank)
            device = torch.device('cuda:%d' % local_rank)
            logger.info('Use GPU %d.' % local_rank)
        else:
            device = torch.device('cuda:%d' % args.gpu)
            torch.cuda.set_device(args.gpu)
            print_rank0('Use GPU %d.' % args.gpu)

    assert args.model in ['mlp', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                          'vgg16', 'vgg19', 'rnn'], 'Model not supported now.'

    assert args.dataset in ['MNIST', 'CIFAR10', 'CIFAR100', 'ImageNet']
    dataset = args.dataset

    if args.model in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'vgg16', 'vgg19'] \
        and args.dataset == 'CIFAR100':
        net = eval(args.model)(100)
    elif args.model == 'rnn':
        net = eval(args.model)(28, 10, 128, 28)
    else:
        net = eval(args.model)()

    assert args.dataset in ['MNIST', 'CIFAR10', 'CIFAR100', 'ImageNet']
    dataset = args.dataset

    net.to(device)
    if args.distributed:
        net = torch.nn.parallel.DistributedDataParallel(
            net, device_ids=[local_rank])

    assert args.opt in ['sgd', 'momentum', 'nesterov',
                        'adagrad', 'adam'], 'Optimizer not supported!'
    if args.opt == 'sgd':
        print_rank0('Use SGD Optimizer.')
        opt = optim.SGD(net.parameters(), lr=args.learning_rate)
    elif args.opt == 'momentum':
        print_rank0('Use Momentum Optimizer.')
        opt = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9)
    elif args.opt == 'nesterov':
        print_rank0('Use Nesterov Momentum Optimizer.')
        opt = optim.SGD(net.parameters(), lr=args.learning_rate,
                        momentum=0.9, nesterov=True)
    elif args.opt == 'adagrad':
        print_rank0('Use AdaGrad Optimizer.')
        opt = optim.Adagrad(net.parameters(), lr=args.learning_rate)
    else:
        print_rank0('Use Adam Optimizer.')
        opt = optim.Adam(lr=args.learning_rate)

    criterion = nn.CrossEntropyLoss()

    # data loading
    print_rank0('Loading %s data...' % dataset)
    if dataset == 'MNIST':
        datasets = ht.data.mnist(onehot=False)
        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]
    elif dataset == 'CIFAR10':
        train_set_x, train_set_y, valid_set_x, valid_set_y = ht.data.normalize_cifar(
            num_class=10, onehot=False)
        if args.model == "mlp":
            train_set_x = train_set_x.reshape(train_set_x.shape[0], -1)
            valid_set_x = valid_set_x.reshape(valid_set_x.shape[0], -1)
    elif dataset == 'CIFAR100':
        train_set_x, train_set_y, valid_set_x, valid_set_y = ht.data.normalize_cifar(
            num_class=100, onehot=False)

    running_time = 0
    # training
    print_rank0("Start training loop...")
    for i in range(args.num_epochs + 1):
        if args.timing:
            start = time()
        train(epoch=i, net=net, data=train_set_x, label=train_set_y,
              batch_size=args.batch_size, criterion=criterion, optimizer=opt)
        if args.timing:
            end = time()
            print_rank0("Running time of current epoch = %fs" % (end - start))
            if i != 0:
                running_time += (end - start)
        if args.validate:
            test(epoch=i, net=net, data=valid_set_x, label=valid_set_y,
                batch_size=args.batch_size, criterion=criterion)

    print_rank0("*"*50)
    print_rank0("Running time of total %d epoch = %fs" %
                (args.num_epochs, running_time))
