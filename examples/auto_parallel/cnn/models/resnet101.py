import hetu as ht
from hetu import layers as htl


def ResNet101():
    def bottleneck(inplanes, outplanes, stride, downsample=htl.Identity()):
        nonlocal layer_id
        sequence = htl.Sequence(
            htl.Conv2d(inplanes, outplanes, kernel_size=1, stride=1,
                       initializer=conv_init, bias=False),
            htl.BatchNorm(outplanes),
            htl.Relu(),

            htl.Conv2d(outplanes, outplanes, kernel_size=3, stride=stride,
                       initializer=conv_init, padding=1, bias=False),
            htl.BatchNorm(outplanes),
            htl.Relu(),

            htl.Conv2d(outplanes, 4 * outplanes, kernel_size=1, stride=1,
                       initializer=conv_init, bias=False),
            htl.BatchNorm(4 * outplanes),
        )
        layer_id += 3
        return lambda x: htl.Relu()(sequence(x) + downsample(x))

    conv_init = ht.init.GenGeneralXavierNormal(gain=2, mode='fan_out')
    global_planes = 64
    layers = [
        htl.Conv2d(3, global_planes, kernel_size=7, stride=2, padding=3,
                   initializer=conv_init, bias=False),
        htl.BatchNorm(global_planes),
        htl.Relu(),
        htl.MaxPool2d(kernel_size=3, stride=2, padding=1),
    ]

    num_blocks = [3, 4, 23, 3]
    all_planes = [64, 128, 256, 512]
    strides = [1, 2, 2, 2]
    layer_id = 2
    for block, planes, stride in zip(num_blocks, all_planes, strides):
        if stride != 1 or global_planes != 4 * planes:
            downsample = htl.Sequence(
                htl.Conv2d(global_planes, 4 * planes, kernel_size=1, stride=stride,
                           initializer=conv_init, bias=False),
                htl.BatchNorm(4 * planes),
            )
        else:
            downsample = htl.Identity()

        layers.append(bottleneck(global_planes, planes, stride, downsample))
        global_planes = 4 * planes
        for _ in range(1, block):
            layers.append(bottleneck(global_planes, planes, 1))

    layers += [
        htl.AvgPool2d(kernel_size=7, stride=1),
        htl.Reshape([-1, 2048]),
        htl.Linear(2048, 1000),
    ]
    return htl.Sequence(*layers)
