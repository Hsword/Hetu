import hetu as ht
from hetu import init


def conv2d(x, in_channel, out_channel, stride=1, padding=1, name=''):
    weight = init.random_normal(
        shape=(out_channel, in_channel, 3, 3), stddev=0.1, name=name+'_weight')
    x = ht.conv2d_op(x, weight, stride=stride, padding=padding)
    return x


def batch_norm_with_relu(x, hidden, name):
    scale = init.random_normal(
        shape=(1, hidden, 1, 1), stddev=0.1, name=name+'_scale')
    bias = init.random_normal(shape=(1, hidden, 1, 1),
                              stddev=0.1, name=name+'_bias')
    x = ht.batch_normalization_op(x, scale, bias)
    x = ht.relu_op(x)
    return x


def resnet_block(x, in_channel, num_blocks, is_first=False, name=''):
    if is_first:
        out_channel = in_channel
        identity = x
        x = conv2d(x, in_channel, out_channel, stride=1,
                   padding=1, name=name+'_conv1')
        x = batch_norm_with_relu(x, out_channel, name+'_bn1')
        x = conv2d(x, out_channel, out_channel, stride=1,
                   padding=1, name=name+'_conv2')
        x = x + identity
    else:
        out_channel = 2 * in_channel
        identity = x
        x = batch_norm_with_relu(x, in_channel, name+'_bn0')
        x = ht.pad_op(x, [[0, 0], [0, 0], [0, 1], [0, 1]])
        x = conv2d(x, in_channel, out_channel, stride=2,
                   padding=0, name=name+'_conv1')
        x = batch_norm_with_relu(x, out_channel, name+'_bn1')
        x = conv2d(x, out_channel, out_channel, stride=1,
                   padding=1, name=name+'_conv2')
        identity = ht.avg_pool2d_op(
            identity, kernel_H=2, kernel_W=2, padding=0, stride=2)
        identity = ht.pad_op(
            identity, [[0, 0], [in_channel // 2, in_channel // 2], [0, 0], [0, 0]])
        x = x + identity

    for i in range(1, num_blocks):
        identity = x
        x = batch_norm_with_relu(x, out_channel, name+'_bn%d' % (2 * i))
        x = conv2d(x, out_channel, out_channel, stride=1,
                   padding=1, name=name+'_conv%d' % (2 * i + 1))
        x = batch_norm_with_relu(x, out_channel, name+'_bn%d' % (2 * i + 1))
        x = conv2d(x, out_channel, out_channel, stride=1,
                   padding=1, name=name+'_conv%d' % (2 * i + 2))
        x = x + identity

    return x


def fc(x, shape, name):
    weight = init.random_normal(shape=shape, stddev=0.1, name=name+'_weight')
    bias = init.random_normal(shape=shape[-1:], stddev=0.1, name=name+'_bias')
    x = ht.matmul_op(x, weight)
    x = x + ht.broadcastto_op(bias, x)
    return x


def resnet(x, y_, num_layers=18, num_class=10):
    '''
    ResNet model, for CIFAR10 dataset.

    Parameters:
        x: Variable(hetu.gpu_ops.Node.Node), shape (N, C, H, W)
        y_: Variable(hetu.gpu_ops.Node.Node), shape (N, num_classes)
        num_layers: 18 or 34
    Return:
        loss: Variable(hetu.gpu_ops.Node.Node), shape (1,)
        y: Variable(hetu.gpu_ops.Node.Node), shape (N, num_classes)
    '''

    base_size = 16

    x = conv2d(x, 3, base_size, stride=1, padding=1,
               name='resnet_initial_conv')
    x = batch_norm_with_relu(x, base_size, 'resnet_initial_bn')

    if num_layers == 18:
        print("Building ResNet-18 model...")
        x = resnet_block(x,     base_size, num_blocks=2,
                         is_first=True, name='resnet_block1')
        x = resnet_block(x,     base_size, num_blocks=2,
                         is_first=False, name='resnet_block2')
        x = resnet_block(x, 2 * base_size, num_blocks=2,
                         is_first=False, name='resnet_block3')
        x = resnet_block(x, 4 * base_size, num_blocks=2,
                         is_first=False, name='resnet_block4')
    elif num_layers == 34:
        print("Building ResNet-34 model...")
        x = resnet_block(x,     base_size, num_blocks=3,
                         is_first=True, name='resnet_block1')
        x = resnet_block(x,     base_size, num_blocks=4,
                         is_first=False, name='resnet_block2')
        x = resnet_block(x, 2 * base_size, num_blocks=6,
                         is_first=False, name='resnet_block3')
        x = resnet_block(x, 4 * base_size, num_blocks=3,
                         is_first=False, name='resnet_block4')
    else:
        assert False, "Number of layers should be 18 or 34 !"

    x = batch_norm_with_relu(x, 8 * base_size, 'resnet_final_bn')
    x = ht.array_reshape_op(x, (-1, 128 * base_size))
    y = fc(x, (128 * base_size, num_class), name='resnet_final_fc')
    # here we don't use cudnn for softmax crossentropy to avoid overflows
    loss = ht.softmaxcrossentropy_op(y, y_, use_cudnn=False)
    loss = ht.reduce_mean_op(loss, [0])
    return loss, y


def resnet18(x, y_, num_class=10):
    return resnet(x, y_, 18, num_class)


def resnet34(x, y_, num_class=10):
    return resnet(x, y_, 34, num_class)
