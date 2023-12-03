from models import AlexNet as hetu_alexnet, VGG19 as hetu_vgg19, ResNet101 as hetu_resnet101, InceptionV3 as hetu_inceptionv3, WideResNet50 as hetu_wresnet50, WideResNet101 as hetu_wresnet101
from torch_models import AlexNet as torch_alexnet, ResNet101 as torch_resnet101, VGG19 as torch_vgg19, InceptionV3 as torch_inceptionv3, wide_resnet50_2 as torch_wresnet50, wide_resnet101_2 as torch_wresnet101
import hetu as ht
import torch
import numpy as np
import argparse


# Please manually disable all the dropout part in models
if __name__ == "__main__":
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        help='model to be used')
    parser.add_argument('--batch-size', type=int,
                        default=2, help='batch size')
    parser.add_argument('--lr', type=float,
                        default=0.001, help='learning rate')
    parser.add_argument('--l2reg', type=float,
                        default=0.0001, help='l2 regularizer')
    parser.add_argument('--test-num', type=int, default=10,
                        help='number of test iterations')
    args = parser.parse_args()

    crop_size = 299 if args.model == 'inception-v3' else 224

    # use synthetic data
    images = [np.random.normal(loc=0.0, scale=0.1, size=(
        args.batch_size, 3, crop_size, crop_size)).astype(np.float32) for _ in range(args.test_num)]
    targets = [np.random.randint(0, 1000, size=(
        args.batch_size,)) for _ in range(args.test_num)]

    # initialize hetu model
    opt = ht.optim.SGDOptimizer(learning_rate=args.lr, l2reg=args.l2reg)
    hetu_model = {
        'alexnet': hetu_alexnet(dropout=0.),
        'vgg19': hetu_vgg19(dropout=0.),
        'resnet101': hetu_resnet101(),
        'inception-v3': hetu_inceptionv3(dropout=0.),
        'wresnet50': hetu_wresnet50(),
        'wresnet101': hetu_wresnet101(),
    }[args.model]
    x = ht.placeholder_op(name='x')
    y_ = ht.placeholder_op(name='y_')
    if args.model != 'inception-v3':
        y = hetu_model(x)
        loss = ht.softmaxcrossentropy_sparse_op(y, y_)
        loss = ht.reduce_mean_op(loss, [0])
    else:
        y, aux = hetu_model(x)
        loss = ht.softmaxcrossentropy_sparse_op(y, y_)
        # loss_aux = ht.softmaxcrossentropy_sparse_op(aux, y_)
        # loss = loss + 0.3 * loss_aux
        loss = ht.reduce_mean_op(loss, [0])
    train_op = opt.minimize(loss)
    executor = ht.Executor([loss, y, train_op], ctx=ht.gpu(0))

    # initialize torch model
    torch_model = {
        'alexnet': torch_alexnet(dropout=0.),
        'resnet101': torch_resnet101(),
        'vgg19': torch_vgg19(dropout=0.),
        'inception-v3': torch_inceptionv3(dropout=0.),
        'wresnet50': torch_wresnet50(),
        'wresnet101': torch_wresnet101(),
    }[args.model].cuda(1)
    torch_loss = torch.nn.CrossEntropyLoss()
    torch_opt = torch.optim.SGD(
        torch_model.parameters(), lr=args.lr, weight_decay=args.l2reg)

    # synchronize parameters
    # with open('torchdict.txt', 'w') as fw:
    #     for param, value in torch_model.state_dict().items():
    #         print(param, value.shape, file=fw, flush=True)
    # exit()
    hetu_dict = {}
    if args.model == 'alexnet':
        cnn_cnt = 0
        mlp_cnt = 0
        visited_cnn = set()
        visited_mlp = set()
        for k, v in torch_model.state_dict().items():
            prefix, idx, param_name = k.split('.')
            idx = int(idx)
            if prefix == 'features':
                if idx not in visited_cnn:
                    cnn_cnt += 1
                    visited_cnn.add(idx)
                hetu_name = 'alexnet_conv{}_{}'.format(cnn_cnt, param_name)
                hetu_dict[hetu_name] = v.cpu().detach().numpy()
            else:
                if idx not in visited_mlp:
                    mlp_cnt += 1
                    visited_mlp.add(idx)
                hetu_name = 'alexnet_fc{}_{}'.format(mlp_cnt, param_name)
                hetu_dict[hetu_name] = v.cpu().detach().numpy().transpose()
    elif args.model in ('resnet101', 'wresnet50', 'wresnet101'):
        # to test resnet101, we need to reduce layer blocks, e.g. [3,3,3,3]
        for k, v in torch_model.state_dict().items():
            if k.split('.')[-1] in ('running_mean', 'running_var', 'num_batches_tracked'):
                continue
            hetu_name = k.replace('.', '_')
            param_value = v.cpu().detach().numpy()
            if hetu_name.startswith('fc'):
                param_value = param_value.transpose()
            hetu_dict[hetu_name] = param_value
    elif args.model == 'vgg19':
        cnn_cnt = 0
        mlp_cnt = 0
        visited_mlp = set()
        for k, v in torch_model.state_dict().items():
            prefix, idx, param_name = k.split('.')
            idx = int(idx)
            if prefix == 'features':
                hetu_name = 'vgg_layer{}_{}'.format(idx, param_name)
                hetu_dict[hetu_name] = v.cpu().detach().numpy()
            else:
                if idx not in visited_mlp:
                    mlp_cnt += 1
                    visited_mlp.add(idx)
                hetu_name = 'vgg_fc{}_{}'.format(mlp_cnt, param_name)
                hetu_dict[hetu_name] = v.cpu().detach().numpy().transpose()
    else:
        # inception-v3
        # to test inception v3, we need to reduce network structure
        for k, v in torch_model.state_dict().items():
            if k.split('.')[-1] in ('running_mean', 'running_var', 'num_batches_tracked'):
                continue
            hetu_name = k.replace('.', '_')
            param_value = v.cpu().detach().numpy()
            layer_type = hetu_name.split('_')[-2]
            if layer_type == 'fc':
                param_value = param_value.transpose()
            hetu_dict[hetu_name] = param_value
    executor.load_dict(hetu_dict)

    # testing
    for i in range(args.test_num):
        image = images[i]
        target = targets[i]

        hetu_loss_val, hetu_predict_y, _ = executor.run(
            feed_dict={x: image, y_: target.astype(np.float32)}, convert_to_numpy_ret_vals=True)
        hetu_loss_val = hetu_loss_val[0]

        torch_opt.zero_grad()
        if args.model == 'inception-v3':
            torch_predict_y, torch_aux_predict_y = torch_model(
                torch.from_numpy(image).cuda(1))
            ground_truth = torch.from_numpy(target).cuda(1)
            torch_loss_val = torch_loss(torch_predict_y, ground_truth)
            # aux_loss = torch_loss(torch_aux_predict_y, ground_truth)
            # torch_loss_val = torch_loss_val + 0.3 * aux_loss
        else:
            torch_predict_y = torch_model(torch.from_numpy(image).cuda(1))
            torch_loss_val = torch_loss(
                torch_predict_y, torch.from_numpy(target).cuda(1))
        torch_loss_val.backward()
        torch_opt.step()
        torch_loss_val = torch_loss_val.cpu().detach().numpy()
        torch_predict_y = torch_predict_y.cpu().detach().numpy()

        np.testing.assert_allclose(
            hetu_predict_y, torch_predict_y, atol=1e-2, rtol=1e-2)
        np.testing.assert_allclose(hetu_loss_val, torch_loss_val, rtol=1e-4)
        print(hetu_loss_val, torch_loss_val)
