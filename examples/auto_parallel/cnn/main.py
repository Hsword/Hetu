from models import AlexNet, VGG19, ResNet101, InceptionV3
import hetu as ht
import os
import numpy as np
import argparse
from time import time
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets


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
    parser.add_argument('--log-num', type=int,
                        default=30, help='number of logs')
    parser.add_argument('--strategy', type=str, default='dp',
                        help='should be dp, mp, owt')
    parser.add_argument('--validate', action='store_true',
                        help='whether to use validation')
    parser.add_argument('--timing', action='store_true',
                        help='whether to time the training phase')
    parser.add_argument('--imagenet', action='store_true',
                        help='whether to use imagenet')
    args = parser.parse_args()

    args.strategy = args.strategy.lower()
    assert args.strategy in ('dp', 'mp', 'owt')

    opt = ht.optim.SGDOptimizer(learning_rate=args.lr, l2reg=args.l2reg)
    model = {
        'alexnet': AlexNet(),
        'vgg19': VGG19(),
        'resnet101': ResNet101(),
        'inception-v3': InceptionV3(),
    }[args.model]
    crop_size = 299 if args.model == 'inception-v3' else 224

    if args.imagenet:
        # use imagenet
        traindir = os.path.join('/home/public/zhl/imagenet', 'train')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_loader = iter(torch.utils.data.DataLoader(
            datasets.ImageFolder(
                traindir, transforms.Compose([
                    transforms.RandomResizedCrop(crop_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])),
            batch_size=args.batch_size, shuffle=True, num_workers=0))
    else:
        # use synthetic data
        images = [np.random.normal(loc=0.0, scale=0.1, size=(
            args.batch_size, 3, crop_size, crop_size)).astype(np.float32) for _ in range(args.log_iterations)]
        targets = [np.random.randint(0, 1000, size=(
            args.batch_size,)).astype(np.float32) for _ in range(args.log_iterations)]

    if args.validate:
        valdir = os.path.join('/home/public/zhl/imagenet', 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False, num_workers=0)

    x = ht.placeholder_op(name='x')
    y_ = ht.placeholder_op(name='y_')

    if args.model != 'inception-v3':
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

    eval_nodes = {'train': [loss, y, train_op],
                  'validate': [loss, y]}
    strategy = {
        'dp': ht.dist.DataParallel(aggregate='allreduce'),
        'mp': ht.dist.ModelParallel4CNN(),
        'owt': ht.dist.OneWeirdTrick4CNN(),
    }[args.strategy]
    executor = ht.Executor(eval_nodes, dist_strategy=strategy)
    printing = executor.rank in (0, None)

    # training
    running_time = 0
    for i in range(args.log_num):
        loss_all = 0
        acc_all = 0
        cnt = 0
        if args.timing:
            start = time()
        while cnt < args.log_iterations:
            if args.imagenet:
                (image, target) = next(train_loader)
                image = image.numpy()
                target = target.numpy().astype(np.float32)
            else:
                image = images[cnt]
                target = targets[cnt]
            loss_val, predict_y, _ = executor.run(
                'train', feed_dict={x: image, y_: target}, convert_to_numpy_ret_vals=True)
            if loss_val is not None:
                loss_all += loss_val[0]
                acc_all += np.sum(np.equal(target, np.argmax(predict_y, 1)))
            cnt += 1
        loss_all, acc_all = list(executor.reduceMean([loss_all, acc_all]))
        if printing:
            loss_all /= args.log_iterations
            acc_all /= args.log_iterations
            print('Train loss = %f' % loss_all)
            print('Train accuracy = %f' % acc_all)
        if args.timing:
            end = time()
            during_time = end - start
            if printing:
                print("Running time of current epoch = %fs" % (during_time))
            if i != 0:
                running_time += during_time
        if args.validate:
            val_loss_all = 0
            val_acc_all = 0
            batch_num = 0
            for (image, target) in val_loader:
                image = image.numpy()
                target = target.numpy()
                loss_val, valid_y_predicted = executor.run(
                    'validate', feed_dict={x: image, y_: target}, convert_to_numpy_ret_vals=True)
                if printing:
                    val_loss_all += loss_val[0]
                    acc_all += np.sum(np.equal(target,
                                               np.argmax(predict_y, 1)))
                batch_num += 1
            if printing:
                val_loss_all /= batch_num
                val_acc_all /= batch_num
                print("Validation loss = %f" % val_loss_all)
                print("Validation accuracy = %f" % val_acc_all)
    if args.timing:
        if printing:
            print("*"*50)
            print("Running time of total %d iterations = %fs" %
                  (args.log_num * args.log_iterations, running_time))
