import hetu as ht
from hetu import layers as htl


def InceptionV3():
    def conv_bn_relu(in_channels, out_channels, kernel_size, stride=1, padding=0, stddev=0.1):
        return htl.Sequence(
            htl.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                       stride=stride, padding=padding,
                       initializer=ht.init.GenTruncatedNormal(stddev=stddev),
                       bias=False),
            htl.BatchNorm(out_channels),
            htl.Relu(),
        )

    def inception_block_a(in_channels, pool_features):
        b1 = conv_bn_relu(in_channels, 64, kernel_size=1)
        b2 = htl.Sequence(
            conv_bn_relu(in_channels, 48, kernel_size=1),
            conv_bn_relu(48, 64, kernel_size=5, padding=2),
        )
        b3 = htl.Sequence(
            conv_bn_relu(in_channels, 64, kernel_size=1),
            conv_bn_relu(64, 96, kernel_size=3, padding=1),
            conv_bn_relu(96, 96, kernel_size=3, padding=1),
        )
        b4 = htl.Sequence(
            htl.AvgPool2d(kernel_size=3, stride=1, padding=1),
            conv_bn_relu(in_channels, pool_features, kernel_size=1),
        )
        return htl.ConcatenateLayers([b1, b2, b3, b4], axis=1)

    def inception_block_b(in_channels):
        b1 = conv_bn_relu(in_channels, 384, kernel_size=3, stride=2)
        b2 = htl.Sequence(
            conv_bn_relu(in_channels, 64, kernel_size=1),
            conv_bn_relu(64, 96, kernel_size=3, padding=1),
            conv_bn_relu(96, 96, kernel_size=3, stride=2),
        )
        b3 = htl.MaxPool2d(kernel_size=3, stride=2)
        return htl.ConcatenateLayers([b1, b2, b3], axis=1)

    def inception_block_c(in_channels, channels7):
        b1 = conv_bn_relu(in_channels, 192, kernel_size=1)
        b2 = htl.Sequence(
            conv_bn_relu(in_channels, channels7, kernel_size=1),
            conv_bn_relu(channels7, channels7,
                         kernel_size=(1, 7), padding=(0, 3)),
            conv_bn_relu(channels7, 192, kernel_size=(7, 1), padding=(3, 0)),
        )
        b3 = htl.Sequence(
            conv_bn_relu(in_channels, channels7, kernel_size=1),
            conv_bn_relu(channels7, channels7,
                         kernel_size=(7, 1), padding=(3, 0)),
            conv_bn_relu(channels7, channels7,
                         kernel_size=(1, 7), padding=(0, 3)),
            conv_bn_relu(channels7, channels7,
                         kernel_size=(7, 1), padding=(3, 0)),
            conv_bn_relu(channels7, 192, kernel_size=(1, 7), padding=(0, 3)),
        )
        b4 = htl.Sequence(
            htl.AvgPool2d(kernel_size=3, stride=1, padding=1),
            conv_bn_relu(in_channels, 192, kernel_size=1),
        )
        return htl.ConcatenateLayers([b1, b2, b3, b4], axis=1)

    def inception_block_d(in_channels):
        b1 = htl.Sequence(
            conv_bn_relu(in_channels, 192, kernel_size=1),
            conv_bn_relu(192, 320, kernel_size=3, stride=2),
        )
        b2 = htl.Sequence(
            conv_bn_relu(in_channels, 192, kernel_size=1),
            conv_bn_relu(192, 192, kernel_size=(1, 7), padding=(0, 3)),
            conv_bn_relu(192, 192, kernel_size=(7, 1), padding=(3, 0)),
            conv_bn_relu(192, 192, kernel_size=3, stride=2),
        )
        b3 = htl.MaxPool2d(kernel_size=3, stride=2)
        return htl.ConcatenateLayers([b1, b2, b3], axis=1)

    def inception_block_e(in_channels):
        b1 = conv_bn_relu(in_channels, 320, kernel_size=1)
        b2 = htl.Sequence(
            conv_bn_relu(in_channels, 384, kernel_size=1),
            htl.ConcatenateLayers([
                conv_bn_relu(384, 384, kernel_size=(1, 3), padding=(0, 1)),
                conv_bn_relu(384, 384, kernel_size=(3, 1), padding=(1, 0)),
            ], axis=1),
        )
        b3 = htl.Sequence(
            conv_bn_relu(in_channels, 448, kernel_size=1),
            conv_bn_relu(448, 384, kernel_size=3, padding=1),
            htl.ConcatenateLayers([
                conv_bn_relu(384, 384, kernel_size=(1, 3), padding=(0, 1)),
                conv_bn_relu(384, 384, kernel_size=(3, 1), padding=(1, 0)),
            ], axis=1),
        )
        b4 = htl.Sequence(
            htl.AvgPool2d(kernel_size=3, stride=1, padding=1),
            conv_bn_relu(in_channels, 192, kernel_size=1),
        )
        return htl.ConcatenateLayers([b1, b2, b3, b4], axis=1)

    def inception_block_aux(in_channels, num_classes):
        return htl.Sequence(
            htl.AvgPool2d(kernel_size=5, stride=3),
            conv_bn_relu(in_channels, 128, kernel_size=1),
            conv_bn_relu(128, 768, kernel_size=5, stddev=0.01),
            htl.Reshape([-1, 768]),
            htl.Linear(768, num_classes,
                       initializer=ht.init.GenTruncatedNormal(stddev=0.001)),
        )

    def result_func(x):
        middle = prev_layers(x)
        return (post_layers(middle), aux_layers(middle))

    prev_layers = htl.Sequence(
        conv_bn_relu(3, 32, kernel_size=3, stride=2),
        conv_bn_relu(32, 32, kernel_size=3),
        conv_bn_relu(32, 64, kernel_size=3, padding=1),
        htl.MaxPool2d(kernel_size=3, stride=2),
        conv_bn_relu(64, 80, kernel_size=1),
        conv_bn_relu(80, 192, kernel_size=3),
        htl.MaxPool2d(kernel_size=3, stride=2),
        inception_block_a(192, 32),
        inception_block_a(256, 64),
        inception_block_a(288, 64),
        inception_block_b(288),
        inception_block_c(768, 128),
        inception_block_c(768, 160),
        inception_block_c(768, 160),
        inception_block_c(768, 192),
    )
    post_layers = htl.Sequence(
        inception_block_d(768),
        inception_block_e(1280),
        inception_block_e(2048),
        htl.AvgPool2d(kernel_size=8, stride=1),
        htl.DropOut(),
        htl.Reshape([-1, 2048]),
        htl.Linear(2048, 1000,
                   initializer=ht.init.GenTruncatedNormal(stddev=0.1)),
    )
    aux_layers = inception_block_aux(768, 1000)

    return result_func
