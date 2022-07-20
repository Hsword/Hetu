from .base import BaseLayer
import hetu as ht


class BatchNorm(BaseLayer):
    def __init__(self, num_channels, scale=True, bias=True, name='batchnorm'):
        self.num_channels = num_channels
        self.name = name
        self.scale_var = ht.init.ones(
            shape=(self.num_channels,), name=self.name+'_weight', trainable=scale)
        self.bias_var = ht.init.zeros(
            shape=(self.num_channels,), name=self.name+'_bias', trainable=bias)

    def __call__(self, x):
        return ht.batch_normalization_op(x, self.scale_var, self.bias_var)


class LayerNorm(BaseLayer):
    def __init__(self, num_channels, scale=True, bias=True, name='layernorm', eps=1e-12):
        self.num_channels = num_channels
        self.name = name
        self.eps = eps
        self.scale_var = ht.init.ones(
            shape=(self.num_channels, ), name=self.name+'_weight', trainable=scale)
        self.bias_var = ht.init.zeros(
            shape=(self.num_channels, ), name=self.name+'_bias', trainable=bias)

    def __call__(self, x):
        return ht.layer_normalization_op(x, self.scale_var, self.bias_var, eps=self.eps)
