import hetu as ht
from hetu import layers as htl


def VGG19(with_bn=True):
    def vgg_block(num_layers, in_channel, out_channel, name):
        layers = []
        for i in range(1, num_layers+1):
            if with_bn:
                layers += [
                    htl.Conv2d(in_channel, out_channel,
                               kernel_size=3, stride=1, padding=1,
                               initializer=ht.init.GenGeneralXavierNormal(
                                   gain=2, mode='fan_out'),
                               name=name+'_layer%d' % i),
                    htl.BatchNorm(out_channel, name=name+'_bnlayer%d' % i),
                    htl.Relu(),
                ]
            else:
                layers.append(htl.Conv2d(in_channel, out_channel,
                                         kernel_size=3, stride=1, padding=1,
                                         initializer=ht.init.GenGeneralXavierNormal(
                                             gain=2, mode='fan_out'),
                                         activation=ht.relu_op,
                                         name=name+'_layer%d' % i))
            in_channel = out_channel
        layers.append(htl.MaxPool2d(kernel_size=2, stride=2, padding=0))
        return layers

    layers = []
    layers += vgg_block(2,   3,  64, 'vgg_block1')
    layers += vgg_block(2,  64, 128, 'vgg_block2')
    layers += vgg_block(4, 128, 256, 'vgg_block3')
    layers += vgg_block(4, 256, 512, 'vgg_block4')
    layers += vgg_block(4, 512, 512, 'vgg_block5')
    layers += [
        htl.Reshape([-1, 512 * 7 * 7]),
        htl.Linear(512 * 7 * 7, 4096,
                   initializer=ht.init.GenNormal(0, 0.01),
                   activation=ht.relu_op, name='vgg_fc1'),
        htl.DropOut(),
        htl.Linear(4096, 4096,
                   initializer=ht.init.GenNormal(0, 0.01),
                   activation=ht.relu_op, name='vgg_fc2'),
        htl.DropOut(),
        htl.Linear(4096, 1000,
                   initializer=ht.init.GenNormal(0, 0.01),
                   name='vgg_fc3'),
    ]
    return htl.Sequence(*layers)
