import hetu as ht


def Conv2d(in_channels, out_channels,
           kernel_size, stride=1, padding=0,
           initializer=ht.init.GenXavierUniform(),
           bias=True, activation=None, name='conv2d'):
    if isinstance(kernel_size, tuple):
        height, width = kernel_size
    else:
        height = width = kernel_size

    def conv2d(x):
        weight_var = initializer(shape=(out_channels, in_channels, height, width),
                                 name=name+'_weight')
        x = ht.conv2d_op(x, weight_var, stride=stride, padding=padding)
        if bias:
            bias_var = ht.init.zeros(
                shape=(1, out_channels, 1, 1), name=name+'_bias')
            x = x + bias_var
        if activation is not None:
            x = activation(x)
        return x
    return conv2d
