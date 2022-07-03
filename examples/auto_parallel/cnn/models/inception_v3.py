import hetu as ht
from hetu import layers as htl


def InceptionV3(dropout=0.5):
    def conv_bn_relu(in_channels, out_channels, kernel_size, stride=1, padding=0, stddev=0.1, name='conv_bn_relu'):
        return htl.Sequence(
            htl.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                       stride=stride, padding=padding,
                       initializer=ht.init.GenTruncatedNormal(stddev=stddev),
                       bias=False, name=name+'_conv'),
            htl.BatchNorm(out_channels, name=name+'_bn'),
            htl.Relu(),
        )

    def inception_block_a(in_channels, pool_features, name='inception_block_a'):
        b1 = conv_bn_relu(in_channels, 64, kernel_size=1,
                          name=name+'_branch1x1')
        b2 = htl.Sequence(
            conv_bn_relu(in_channels, 48, kernel_size=1,
                         name=name+'_branch5x5_1'),
            conv_bn_relu(48, 64, kernel_size=5, padding=2,
                         name=name+'_branch5x5_2'),
        )
        b3 = htl.Sequence(
            conv_bn_relu(in_channels, 64, kernel_size=1,
                         name=name+'_branch3x3dbl_1'),
            conv_bn_relu(64, 96, kernel_size=3, padding=1,
                         name=name+'_branch3x3dbl_2'),
            conv_bn_relu(96, 96, kernel_size=3, padding=1,
                         name=name+'_branch3x3dbl_3'),
        )
        b4 = htl.Sequence(
            htl.AvgPool2d(kernel_size=3, stride=1, padding=1),
            conv_bn_relu(in_channels, pool_features,
                         kernel_size=1, name=name+'_branch_pool'),
        )
        return htl.ConcatenateLayers([b1, b2, b3, b4], axis=1)

    def inception_block_b(in_channels, name='inception_block_b'):
        b1 = conv_bn_relu(in_channels, 384, kernel_size=3,
                          stride=2, name=name+'_branch3x3')
        b2 = htl.Sequence(
            conv_bn_relu(in_channels, 64, kernel_size=1,
                         name=name+'_branch3x3dbl_1'),
            conv_bn_relu(64, 96, kernel_size=3, padding=1,
                         name=name+'_branch3x3dbl_2'),
            conv_bn_relu(96, 96, kernel_size=3, stride=2,
                         name=name+'_branch3x3dbl_3'),
        )
        b3 = htl.MaxPool2d(kernel_size=3, stride=2)
        return htl.ConcatenateLayers([b1, b2, b3], axis=1)

    def inception_block_c(in_channels, channels7, name='inception_block_c'):
        b1 = conv_bn_relu(in_channels, 192, kernel_size=1,
                          name=name+'_branch1x1')
        b2 = htl.Sequence(
            conv_bn_relu(in_channels, channels7, kernel_size=1,
                         name=name+'_branch7x7_1'),
            conv_bn_relu(channels7, channels7,
                         kernel_size=(1, 7), padding=(0, 3), name=name+'_branch7x7_2'),
            conv_bn_relu(channels7, 192, kernel_size=(7, 1),
                         padding=(3, 0), name=name+'_branch7x7_3'),
        )
        b3 = htl.Sequence(
            conv_bn_relu(in_channels, channels7, kernel_size=1,
                         name=name+'_branch7x7dbl_1'),
            conv_bn_relu(channels7, channels7,
                         kernel_size=(7, 1), padding=(3, 0), name=name+'_branch7x7dbl_2'),
            conv_bn_relu(channels7, channels7,
                         kernel_size=(1, 7), padding=(0, 3), name=name+'_branch7x7dbl_3'),
            conv_bn_relu(channels7, channels7,
                         kernel_size=(7, 1), padding=(3, 0), name=name+'_branch7x7dbl_4'),
            conv_bn_relu(channels7, 192, kernel_size=(1, 7),
                         padding=(0, 3), name=name+'_branch7x7dbl_5'),
        )
        b4 = htl.Sequence(
            htl.AvgPool2d(kernel_size=3, stride=1, padding=1),
            conv_bn_relu(in_channels, 192, kernel_size=1,
                         name=name+'_branch_pool'),
        )
        return htl.ConcatenateLayers([b1, b2, b3, b4], axis=1)

    def inception_block_d(in_channels, name='inception_block_d'):
        b1 = htl.Sequence(
            conv_bn_relu(in_channels, 192, kernel_size=1,
                         name=name+'_branch3x3_1'),
            conv_bn_relu(192, 320, kernel_size=3, stride=2,
                         name=name+'_branch3x3_2'),
        )
        b2 = htl.Sequence(
            conv_bn_relu(in_channels, 192, kernel_size=1,
                         name=name+'_branch7x7x3_1'),
            conv_bn_relu(192, 192, kernel_size=(1, 7), padding=(
                0, 3), name=name+'_branch7x7x3_2'),
            conv_bn_relu(192, 192, kernel_size=(7, 1), padding=(
                3, 0), name=name+'_branch7x7x3_3'),
            conv_bn_relu(192, 192, kernel_size=3, stride=2,
                         name=name+'_branch7x7x3_4'),
        )
        b3 = htl.MaxPool2d(kernel_size=3, stride=2)
        return htl.ConcatenateLayers([b1, b2, b3], axis=1)

    def inception_block_e(in_channels, name='inception_block_e'):
        b1 = conv_bn_relu(in_channels, 320, kernel_size=1,
                          name=name+'_branch1x1')
        b2 = htl.Sequence(
            conv_bn_relu(in_channels, 384, kernel_size=1,
                         name=name+'_branch3x3_1'),
            htl.ConcatenateLayers([
                conv_bn_relu(384, 384, kernel_size=(1, 3),
                             padding=(0, 1), name=name+'_branch3x3_2a'),
                conv_bn_relu(384, 384, kernel_size=(3, 1),
                             padding=(1, 0), name=name+'_branch3x3_2b'),
            ], axis=1),
        )
        b3 = htl.Sequence(
            conv_bn_relu(in_channels, 448, kernel_size=1,
                         name=name+'_branch3x3dbl_1'),
            conv_bn_relu(448, 384, kernel_size=3, padding=1,
                         name=name+'_branch3x3dbl_2'),
            htl.ConcatenateLayers([
                conv_bn_relu(384, 384, kernel_size=(1, 3), padding=(
                    0, 1), name=name+'_branch3x3dbl_3a'),
                conv_bn_relu(384, 384, kernel_size=(3, 1), padding=(
                    1, 0), name=name+'_branch3x3dbl_3b'),
            ], axis=1),
        )
        b4 = htl.Sequence(
            htl.AvgPool2d(kernel_size=3, stride=1, padding=1),
            conv_bn_relu(in_channels, 192, kernel_size=1,
                         name=name+'_branch_pool'),
        )
        return htl.ConcatenateLayers([b1, b2, b3, b4], axis=1)

    def inception_block_aux(in_channels, num_classes, name='inception_block_aux'):
        return htl.Sequence(
            htl.AvgPool2d(kernel_size=5, stride=3),
            conv_bn_relu(in_channels, 128, kernel_size=1, name=name+'_conv0'),
            conv_bn_relu(128, 768, kernel_size=5,
                         stddev=0.01, name=name+'_conv1'),
            htl.Reshape([-1, 768]),
            htl.Linear(768, num_classes,
                       initializer=ht.init.GenTruncatedNormal(stddev=0.001), name=name+'_fc'),
        )

    def result_func(x):
        middle = prev_layers(x)
        return (post_layers(middle), aux_layers(middle))

    prev_layers = htl.Sequence(
        conv_bn_relu(3, 32, kernel_size=3, stride=2, name='Conv2d_1a_3x3'),
        conv_bn_relu(32, 32, kernel_size=3, name='Conv2d_2a_3x3'),
        conv_bn_relu(32, 64, kernel_size=3, padding=1, name='Conv2d_2b_3x3'),
        htl.MaxPool2d(kernel_size=3, stride=2),
        conv_bn_relu(64, 80, kernel_size=1, name='Conv2d_3b_1x1'),
        conv_bn_relu(80, 192, kernel_size=3, name='Conv2d_4a_3x3'),
        htl.MaxPool2d(kernel_size=3, stride=2),
        inception_block_a(192, 32, name='Mixed_5b'),
        inception_block_a(256, 64, name='Mixed_5c'),
        inception_block_a(288, 64, name='Mixed_5d'),
        inception_block_b(288, name='Mixed_6a'),
        inception_block_c(768, 128, name='Mixed_6b'),
        inception_block_c(768, 160, name='Mixed_6c'),
        inception_block_c(768, 160, name='Mixed_6d'),
        inception_block_c(768, 192, name='Mixed_6e'),
    )
    post_layers = htl.Sequence(
        inception_block_d(768, name='Mixed_7a'),
        inception_block_e(1280, name='Mixed_7b'),
        inception_block_e(2048, name='Mixed_7c'),
        htl.AvgPool2d(kernel_size=8, stride=1),
        htl.DropOut(dropout),
        htl.Reshape([-1, 2048]),
        htl.Linear(2048, 1000,
                   initializer=ht.init.GenTruncatedNormal(stddev=0.1), name='fc'),
    )
    aux_layers = inception_block_aux(768, 1000, name='AuxLogits')

    return result_func
