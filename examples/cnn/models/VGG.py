import hetu as ht
from hetu import init


def conv_bn_relu(x, in_channel, out_channel, name):
    weight = init.he_normal(shape=(out_channel, in_channel, 3, 3), name=name+'_weight')
    bn_scale = init.ones(shape=(out_channel,), name=name+'_bn_scale')
    bn_bias = init.zeros(shape=(out_channel,), name=name+'_bn_bias')

    x = ht.conv2d_op(x, weight, padding=1, stride=1)
    x = ht.batch_normalization_op(x, bn_scale, bn_bias)
    act = ht.relu_op(x)
    return act


def vgg_2block(x, in_channel, out_channel, name):
    x = conv_bn_relu(x, in_channel, out_channel, name=name+'_layer1')
    x = conv_bn_relu(x, out_channel, out_channel, name=name+'_layer2')
    x = ht.max_pool2d_op(x, kernel_H=2, kernel_W=2, padding=0, stride=2)
    return x


def vgg_3block(x, in_channel, out_channel, name):
    x = conv_bn_relu(x, in_channel, out_channel, name=name+'_layer1')
    x = conv_bn_relu(x, out_channel, out_channel, name=name+'_layer2')
    x = conv_bn_relu(x, out_channel, out_channel, name=name+'_layer3')
    x = ht.max_pool2d_op(x, kernel_H=2, kernel_W=2, padding=0, stride=2)
    return x


def vgg_4block(x, in_channel, out_channel, name):
    x = conv_bn_relu(x, in_channel, out_channel, name=name+'_layer1')
    x = conv_bn_relu(x, out_channel, out_channel, name=name+'_layer2')
    x = conv_bn_relu(x, out_channel, out_channel, name=name+'_layer3')
    x = conv_bn_relu(x, out_channel, out_channel, name=name+'_layer4')
    x = ht.max_pool2d_op(x, kernel_H=2, kernel_W=2, padding=0, stride=2)
    return x


def vgg_fc(x, in_feat, out_feat, name):
    weight = init.xavier_uniform(shape=(in_feat, out_feat), name=name+'_weight')
    bias = init.zeros(shape=(out_feat,), name=name+'_bias')
    x = ht.matmul_op(x, weight)
    x = x + ht.broadcastto_op(bias, x)
    return x


def vgg(x, y_, num_layers, num_class=10):
    '''
    VGG model, for CIFAR10/CIFAR100 dataset.

    Parameters:
        x: Variable(hetu.gpu_ops.Node.Node), shape (N, C, H, W)
        y_: Variable(hetu.gpu_ops.Node.Node), shape (N, num_classes)
        num_layers: 16 or 19
    Return:
        loss: Variable(hetu.gpu_ops.Node.Node), shape (1,)
        y: Variable(hetu.gpu_ops.Node.Node), shape (N, num_classes)
    '''

    if num_layers == 16:
        print('Building VGG-16 model...')
        x = vgg_2block(x,   3,  64, 'vgg_block1')
        x = vgg_2block(x,  64, 128, 'vgg_block2')
        x = vgg_3block(x, 128, 256, 'vgg_block3')
        x = vgg_3block(x, 256, 512, 'vgg_block4')
        x = vgg_3block(x, 512, 512, 'vgg_block5')

    elif num_layers == 19:
        print('Building VGG-19 model...')
        x = vgg_2block(x,   3,  64, 'vgg_block1')
        x = vgg_2block(x,  64, 128, 'vgg_block2')
        x = vgg_4block(x, 128, 256, 'vgg_block3')
        x = vgg_4block(x, 256, 512, 'vgg_block4')
        x = vgg_4block(x, 512, 512, 'vgg_block5')

    else:
        assert False, 'VGG model should have 16 or 19 layers!'

    x = ht.array_reshape_op(x, (-1, 512))
    x = vgg_fc(x,  512, 4096, 'vgg_fc1')
    x = vgg_fc(x, 4096, 4096, 'vgg_fc2')
    y = vgg_fc(x, 4096, num_class, 'vgg_fc3')
    loss = ht.softmaxcrossentropy_op(y, y_)
    loss = ht.reduce_mean_op(loss, [0])

    return loss, y


def vgg16(x, y_, num_class=10):
    return vgg(x, y_, 16, num_class)


def vgg19(x, y_, num_class=10):
    return vgg(x, y_, 19, num_class)
