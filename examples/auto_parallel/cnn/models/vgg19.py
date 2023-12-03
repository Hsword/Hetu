import hetu as ht
from hetu import layers as htl


def VGG19(with_bn=True, dropout=0.5):
    cfg = [64, 64, "M", 128, 128, "M", 256, 256, 256, 256,
           "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"]
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [htl.MaxPool2d(kernel_size=2, stride=2, padding=0)]
        else:
            cur_len = len(layers)
            if with_bn:
                layers += [
                    htl.Conv2d(in_channels, v,
                               kernel_size=3, stride=1, padding=1,
                               initializer=ht.init.GenGeneralXavierNormal(
                                   gain=2, mode='fan_out'),
                               name='vgg_layer%d' % cur_len),
                    htl.BatchNorm(v, name='vgg_layer%d' % (cur_len + 1)),
                    htl.Relu(),
                ]
            else:
                layers += [htl.Conv2d(in_channels, v,
                                      kernel_size=3, stride=1, padding=1,
                                      initializer=ht.init.GenGeneralXavierNormal(
                                          gain=2, mode='fan_out'),
                                      activation=ht.relu_op,
                                      name='vgg_layer%d' % cur_len)]
            in_channels = v
    layers += [
        htl.Reshape([-1, 512 * 7 * 7]),
        htl.Linear(512 * 7 * 7, 4096,
                   initializer=ht.init.GenNormal(0, 0.01),
                   activation=ht.relu_op, name='vgg_fc1'),
        htl.DropOut(dropout),
        htl.Linear(4096, 4096,
                   initializer=ht.init.GenNormal(0, 0.01),
                   activation=ht.relu_op, name='vgg_fc2'),
        htl.DropOut(dropout),
        htl.Linear(4096, 1000,
                   initializer=ht.init.GenNormal(0, 0.01),
                   name='vgg_fc3'),
    ]
    return htl.Sequence(*layers)
