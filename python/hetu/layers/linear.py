import hetu as ht


def Linear(in_features, out_features,
           initializer=ht.init.GenXavierUniform(),
           bias=True, activation=None, name='linear'):
    def linear(x):
        weight_var = initializer(
            shape=(in_features, out_features), name=name+'_weight')
        x = ht.matmul_op(x, weight_var)
        if bias:
            bias_var = ht.init.zeros(shape=(out_features,), name=name+'_bias')
            x = x + bias_var
        if activation is not None:
            x = activation(x)
        return x
    return linear
