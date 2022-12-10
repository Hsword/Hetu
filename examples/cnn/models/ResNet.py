import hetu as ht
from hetu import init
from hetu import ndarray

def conv2d(x, in_channel, out_channel, stride=1, padding=1, kernel_size=3, name=''):
    a = torch.nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride)
    weight = ht.Variable(name=name+'_weight', value=a.weight.detach().numpy(), ctx=x.ctx)
    # weight = init.he_normal(
    #     shape=(out_channel, in_channel, kernel_size, kernel_size), name=name+'_weight')
    x = ht.conv2d_op(x, weight, stride=stride, padding=padding)
    return x


def batch_norm_with_relu(x, hidden, name):
    scale = init.ones(shape=(hidden,), name=name+'_scale')
    bias = init.zeros(shape=(hidden,), name=name+'_bias')
    x = ht.batch_normalization_op(x, scale, bias, momentum=0.9, eps=1e-5)
    x = ht.relu_op(x)
    return x

def batch_norm(x, hidden, name):
    scale = init.ones(shape=(hidden,), name=name+'_scale')
    bias = init.zeros(shape=(hidden,), name=name+'_bias')
    x = ht.batch_normalization_op(x, scale, bias, momentum=0.9, eps=1e-5)
    return x

def bottleneck(x, input_channel, channel, stride=1, name=''):
    # bottleneck architecture used when layer > 34
    # there are 3 block in reset that should set stride to 2
    # when channel expands, use 11 conv to expand identity
    output_channel = 4 * channel
    shortcut = x
    x = conv2d(x, input_channel, channel, stride=stride, kernel_size=1, padding=0, name=name+'_conv11a')
    x = batch_norm_with_relu(x, channel, name+'_bn1')

    x = conv2d(x, channel, channel, kernel_size=3, padding=1, name=name+'_conv33')
    x = batch_norm_with_relu(x, channel, name+'_bn2')

    x = conv2d(x, channel, output_channel, kernel_size=1, padding=0, name=name+'_conv11b')
    x = batch_norm(x, output_channel, name+'_bn2')

    if input_channel != output_channel:
        shortcut = conv2d(shortcut, input_channel, output_channel,
            kernel_size=1, stride=stride, padding=0, name=name+'_conv11c')
        shortcut = batch_norm(shortcut, output_channel, name+'_bn3')

    x = x + shortcut
    x = ht.relu_op(x)

    return x, output_channel

def basic_block(x, input_channel, output_channel, stride=1, name=''):
    # there are 3 block in reset that should set stride to 2
    # when channel expands, use 11 conv to expand identity
    shortcut = x
    x = conv2d(x, input_channel, output_channel, stride=stride, kernel_size=3, name=name+'_conv33a')
    x = batch_norm_with_relu(x, output_channel, name+'_bn1')

    x = conv2d(x, output_channel, output_channel, stride=1, kernel_size=3, name=name+'_conv33b')
    x = batch_norm(x, output_channel, name+'_bn2')

    if input_channel != output_channel or stride > 1:
        shortcut = conv2d(shortcut, input_channel, output_channel,
            kernel_size=1, stride=stride, padding=0, name=name+'_conv11')
        shortcut = batch_norm(shortcut, output_channel, name+'_bn3')

    x = x + shortcut
    x = ht.relu_op(x)

    return x, output_channel

def fc(x, shape, name):
    weight = init.he_normal(shape=shape, name=name+'_weight')
    bias = init.zeros(shape=shape[-1:], name=name+'_bias')
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

    cur_channel = 16

    x = conv2d(x, 3, cur_channel, stride=1, padding=1,
               name='resnet_initial_conv')

    x = batch_norm_with_relu(x, cur_channel, 'resnet_initial_bn')

    channels = [16, 32, 64, 128]

    if num_layers == 18:
        layers = [2, 2, 2, 2]
    elif num_layers == 34:
        layers = [3, 4, 6, 3]
    elif num_layers == 50:
        layers = [3, 4, 6, 3]
    elif num_layers == 101:
        layers = [3, 4, 23, 3]
    elif num_layers == 152:
        layers = [3, 8, 36, 3]
    else:
        assert False

    if num_layers > 34:
        block = bottleneck
    else:
        block = basic_block

    for i in range(len(layers)):
        for k in range(layers[i]):
            stride = 2 if k == 0 and i > 0 else 1
            x, cur_channel = block(
                x, cur_channel, channels[i], stride=stride,
                name='resnet_block_{}_{}'.format(i, k)
            )

    x = ht.reduce_mean_op(x, [2, 3]) # H, W
    y = fc(x, (cur_channel, num_class), name='resnet_final_fc')
    # here we don't use cudnn for softmax crossentropy to avoid overflows
    loss = ht.softmaxcrossentropy_op(y, y_, use_cudnn=True)
    loss = ht.reduce_mean_op(loss, [0])
    return loss, y


def resnet18(x, y_, num_class=10):
    return resnet(x, y_, 18, num_class)

def resnet34(x, y_, num_class=10):
    return resnet(x, y_, 34, num_class)

def resnet50(x, y_, num_class=10):
    return resnet(x, y_, 50, num_class)

def resnet101(x, y_, num_class=10):
    return resnet(x, y_, 101, num_class)

def resnet152(x, y_, num_class=10):
    return resnet(x, y_, 152, num_class)
