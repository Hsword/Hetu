from .base import BaseLayer
import hetu as ht


class Linear(BaseLayer):
    def __init__(self, in_features, out_features,
                 initializer=ht.init.GenXavierUniform(),
                 bias=True, activation=None, name='linear'):
        self.in_features = in_features
        self.out_features = out_features
        self.initializer = initializer
        self.bias = bias
        if isinstance(activation, str):
            if activation == 'relu':
                activation = ht.relu_op
            else:
                raise NotImplementedError
        self.activation = activation
        self.name = name

    def __call__(self, x):
        weight_var = self.initializer(
            shape=(self.in_features, self.out_features), name=self.name+'_weight')
        x = ht.matmul_op(x, weight_var)
        if self.bias:
            bias_var = ht.init.zeros(
                shape=(self.out_features,), name=self.name+'_bias')
            x = x + bias_var
        if self.activation is not None:
            x = self.activation(x)
        return x
