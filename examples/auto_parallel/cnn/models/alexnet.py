import hetu as ht
from hetu import layers as htl


def AlexNet(dropout=0.5, bias=True):
    return htl.Sequence(
        # conv2d-1
        htl.Conv2d(3, 64, kernel_size=11, stride=4, padding=2,
                   activation=ht.relu_op, name='alexnet_conv1', bias=bias),
        htl.MaxPool2d(kernel_size=3, stride=2, padding=0),
        # conv2d-2
        htl.Conv2d(64, 192, kernel_size=5, stride=1, padding=2,
                   activation=ht.relu_op, name='alexnet_conv2', bias=bias),
        htl.MaxPool2d(kernel_size=3, stride=2, padding=0),
        # conv2d-3
        htl.Conv2d(192, 384, kernel_size=3, stride=1, padding=1,
                   activation=ht.relu_op, name='alexnet_conv3', bias=bias),
        # conv2d-4
        htl.Conv2d(384, 256, kernel_size=3, stride=1, padding=1,
                   activation=ht.relu_op, name='alexnet_conv4', bias=bias),
        # conv2d-5
        htl.Conv2d(256, 256, kernel_size=3, stride=1, padding=1,
                   activation=ht.relu_op, name='alexnet_conv5', bias=bias),
        htl.MaxPool2d(kernel_size=3, stride=2, padding=0),

        # linear
        htl.Reshape([-1, 256 * 6 * 6]),
        htl.DropOut(dropout),
        htl.Linear(256 * 6 * 6, 4096, activation=ht.relu_op,
                   name='alexnet_fc1', bias=bias),
        htl.DropOut(dropout),
        htl.Linear(4096, 4096, activation=ht.relu_op,
                   name='alexnet_fc2', bias=bias),
        htl.Linear(4096, 1000, name='alexnet_fc3', bias=bias),
    )
