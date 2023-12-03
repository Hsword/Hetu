import hetu as ht
import hetu.layers as htl


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, name='conv3*3'):
    """3x3 convolution with padding"""
    return htl.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
        name=name,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, name='conv1*1'):
    """1x1 convolution"""
    return htl.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, name=name)


class Bottleneck(object):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample=None,
        base_width: int = 64,
        norm_layer=None,
        name='bottleneck',
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = htl.BatchNorm
        width = int(planes * (base_width / 64.0))
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width, name=name+'_conv1')
        self.bn1 = norm_layer(width, name=name+'_bn1')
        self.conv2 = conv3x3(width, width, stride, name=name+'_conv2')
        self.bn2 = norm_layer(width, name=name+'_bn2')
        self.conv3 = conv1x1(
            width, planes * self.expansion, name=name+'_conv3')
        self.bn3 = norm_layer(planes * self.expansion, name=name+'_bn3')
        self.relu = htl.Relu()
        self.downsample = downsample
        self.stride = stride

    def __call__(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = ht.sum_op([out, identity])
        out = self.relu(out)

        return out


class ResNet(object):
    def __init__(
        self,
        layers,
        num_classes: int = 1000,
        width_per_group: int = 64,
    ) -> None:
        super().__init__()
        norm_layer = htl.BatchNorm
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.base_width = width_per_group
        self.conv1 = htl.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False, name='conv1')
        self.bn1 = norm_layer(self.inplanes, name='bn1')
        self.relu = htl.Relu()
        self.maxpool = htl.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, layers[0], name='layer1')
        self.layer2 = self._make_layer(128, layers[1], stride=2, name='layer2')
        self.layer3 = self._make_layer(256, layers[2], stride=2, name='layer3')
        self.layer4 = self._make_layer(512, layers[3], stride=2, name='layer4')
        self.avgpool = htl.AvgPool2d(kernel_size=7, stride=1)
        self.reshape = htl.Reshape((-1, 512 * Bottleneck.expansion))
        self.fc = htl.Linear(512 * Bottleneck.expansion,
                             num_classes, name='fc')

    def _make_layer(
        self,
        planes: int,
        blocks: int,
        stride: int = 1,
        name: str = 'layer',
    ):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = htl.Sequence(
                conv1x1(self.inplanes, planes * Bottleneck.expansion,
                        stride, name=name+'_0_downsample_0'),
                norm_layer(planes * Bottleneck.expansion,
                           name+'_0_downsample_1'),
            )

        layers = []
        layers.append(
            Bottleneck(
                self.inplanes, planes, stride, downsample, self.base_width, norm_layer, name=name+'_0'
            )
        )
        self.inplanes = planes * Bottleneck.expansion
        for i in range(1, blocks):
            layers.append(
                Bottleneck(
                    self.inplanes,
                    planes,
                    base_width=self.base_width,
                    norm_layer=norm_layer,
                    name=name+'_{}'.format(i)
                )
            )

        return htl.Sequence(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.reshape(x)
        x = self.fc(x)

        return x

    def __call__(self, x):
        return self._forward_impl(x)


def WideResNet50(**kwargs) -> ResNet:
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["width_per_group"] = 64 * 2
    return ResNet([3, 4, 6, 3], **kwargs)


def WideResNet101(**kwargs) -> ResNet:
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["width_per_group"] = 64 * 2
    return ResNet([3, 4, 23, 3], **kwargs)
