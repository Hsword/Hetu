import hetu as ht
from hetu import layers as htl


def ResNet101():
    def bottleneck(inplanes, outplanes, stride, downsample=htl.Identity(), name='bottleneck'):
        nonlocal layer_id
        main_sequence = htl.Sequence(
            htl.Conv2d(inplanes, outplanes, kernel_size=1, stride=1,
                       initializer=conv_init, bias=False, name=name+'_conv1'),
            htl.BatchNorm(outplanes, name=name+'_bn1'),
            htl.Relu(),

            htl.Conv2d(outplanes, outplanes, kernel_size=3, stride=stride,
                       initializer=conv_init, padding=1, bias=False, name=name+'_conv2'),
            htl.BatchNorm(outplanes, name=name+'_bn2'),
            htl.Relu(),

            htl.Conv2d(outplanes, 4 * outplanes, kernel_size=1, stride=1,
                       initializer=conv_init, bias=False, name=name+'_conv3'),
            htl.BatchNorm(4 * outplanes, name=name+'_bn3'),
        )
        sequence = htl.Sequence(
            htl.SumLayers([main_sequence, downsample]),
            htl.Relu(),
        )
        layer_id += 3
        return sequence

    conv_init = ht.init.GenGeneralXavierNormal(gain=2, mode='fan_out')
    global_planes = 64
    layers = [
        htl.Conv2d(3, global_planes, kernel_size=7, stride=2, padding=3,
                   initializer=conv_init, bias=False, name='conv1'),
        htl.BatchNorm(global_planes, name='bn1'),
        htl.Relu(),
        htl.MaxPool2d(kernel_size=3, stride=2, padding=1),
    ]

    num_blocks = [3, 4, 23, 3]
    all_planes = [64, 128, 256, 512]
    strides = [1, 2, 2, 2]
    layer_id = 2
    for i, (block, planes, stride) in enumerate(zip(num_blocks, all_planes, strides)):
        if stride != 1 or global_planes != 4 * planes:
            downsample = htl.Sequence(
                htl.Conv2d(global_planes, 4 * planes, kernel_size=1, stride=stride,
                           initializer=conv_init, bias=False, name='layer{}_0_downsample_0'.format(i+1)),
                htl.BatchNorm(
                    4 * planes, name='layer{}_0_downsample_1'.format(i+1)),
            )
        else:
            assert False, 'Not allowed in ResNet101!'
            downsample = htl.Identity()

        layers.append(bottleneck(global_planes, planes, stride,
                      downsample, name='layer{}_{}'.format(i+1, 0)))
        global_planes = 4 * planes
        for j in range(1, block):
            layers.append(bottleneck(global_planes, planes, 1,
                          name='layer{}_{}'.format(i+1, j)))

    layers += [
        htl.AvgPool2d(kernel_size=7, stride=1),
        htl.Reshape([-1, 2048]),
        htl.Linear(2048, 1000, name='fc'),
    ]
    return htl.Sequence(*layers)
