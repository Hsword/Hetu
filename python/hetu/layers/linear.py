from .base import BaseLayer
from ..gpu_ops.Node import Op
import hetu as ht


class Linear(BaseLayer):
    def __init__(self, in_features, out_features, initializer=ht.init.GenXavierUniform(),
                 bias=True, activation=None, weight_transpose=False, name='linear'):
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        if isinstance(activation, str):
            if activation == 'relu':
                activation = ht.relu_op
            else:
                raise NotImplementedError
        self.activation = activation
        self.weight_transpose = weight_transpose
        self.name = name
        if isinstance(initializer, Op):
            # in case users want to pass in the weight
            self.weight_var = initializer
        else:
            weight_shape = (self.out_features, self.in_features) if weight_transpose else (
                self.in_features, self.out_features)
            self.weight_var = initializer(
                shape=weight_shape, name=self.name+'_weight')
        if self.bias:
            self.bias_var = ht.init.zeros(
                shape=(self.out_features,), name=self.name+'_bias')

    def __call__(self, x):
        if self.bias:
            x = ht.linear_op(x, self.weight_var, self.bias_var,
                             trans_B=self.weight_transpose)
        else:
            x = ht.matmul_op(x, self.weight_var, trans_B=self.weight_transpose)
        if self.activation is not None:
            x = self.activation(x)
        return x
