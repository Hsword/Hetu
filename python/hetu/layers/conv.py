from .base import BaseLayer
import hetu as ht


class Conv2d(BaseLayer):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride=1, padding=0,
                 initializer=ht.init.GenXavierUniform(),
                 bias=True, activation=None, name='conv2d'):

        if isinstance(kernel_size, tuple):
            height, width = kernel_size
        else:
            height = width = kernel_size
        self.height = height
        self.width = width
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.initializer = initializer
        self.bias = bias
        self.activation = activation
        self.name = name

    def __call__(self, x):
        weight_var = self.initializer(shape=(self.out_channels, self.in_channels, self.height, self.width),
                                      name=self.name+'_weight')
        if self.bias:
            bias_var = ht.init.zeros(
                shape=(self.out_channels,), name=self.name+'_bias')
            x = ht.conv2d_add_bias_op(
                x, weight_var, bias_var, stride=self.stride, padding=self.padding)
        else:
            x = ht.conv2d_op(x, weight_var, stride=self.stride,
                             padding=self.padding)
        if self.activation is not None:
            x = self.activation(x)
        return x
