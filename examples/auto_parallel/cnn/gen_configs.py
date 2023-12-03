from models import AlexNet, VGG19, ResNet101, InceptionV3
import hetu as ht
import os
import argparse


if __name__ == "__main__":
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        help='model to be used')
    parser.add_argument('--save-dir', type=str, default='test_strategy',
                        help='save directory')
    parser.add_argument('--count', type=int, default=3,
                        help='sample number')
    parser.add_argument('--test', action='store_true',
                        help='whether used for test correctness')
    parser.add_argument('--pipedream', action='store_true',
                        help='whether for pipedream or not')
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    opt = ht.optim.SGDOptimizer(learning_rate=0.001)
    if args.test:
        model = AlexNet(dropout=0., bias=False)
    else:
        if args.model == 'alexnet':
            model = AlexNet()
        elif args.model == 'vgg19':
            model = VGG19()
        elif args.model == 'resnet101':
            model = ResNet101()
        elif args.model == 'inception-v3':
            model = InceptionV3()
        else:
            assert False
    crop_size = 299 if args.model == 'inception-v3' else 224
    global_batch_size = 128

    x = ht.placeholder_op(name='x')
    y_ = ht.placeholder_op(name='y_')

    if args.model == 'inception-v3':
        y, aux = model(x)
        loss = ht.softmaxcrossentropy_sparse_op(y, y_)
        loss_aux = ht.softmaxcrossentropy_sparse_op(aux, y_)
        loss = loss + 0.3 * loss_aux
    else:
        y = model(x)
        loss = ht.softmaxcrossentropy_sparse_op(y, y_)
    loss = ht.reduce_mean_op(loss, [0])
    train_op = opt.minimize(loss)
    eval_nodes = [loss, y, train_op]

    strategy = ht.dist.BaseSearchingStrategy(
        {x: (global_batch_size, 3, crop_size, crop_size), y_: (global_batch_size,)}, include_duplicate=False)
    strategy.generate_random_config(eval_nodes, os.path.join(
        args.save_dir, 'config{}.json'), count=args.count, fixed_loss=args.test, pipedream=args.pipedream)
