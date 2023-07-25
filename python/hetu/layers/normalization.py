from .base import BaseLayer
import hetu as ht


class BatchNorm(BaseLayer):
    def __init__(self, num_channels, name='batchnorm'):
        self.num_channels = num_channels
        self.name = name
        self.scale_var = ht.init.ones(
            shape=(self.num_channels,), name=self.name+'.weight')
        self.bias_var = ht.init.zeros(
            shape=(self.num_channels,), name=self.name+'.bias')

    def __call__(self, x):
        return ht.batch_normalization_op(x, self.scale_var, self.bias_var)


class LayerNorm(BaseLayer):
    def __init__(self, num_channels, name='layernorm', eps=1e-05):
        self.num_channels = num_channels
        self.name = name
        self.eps = eps
        self.scale_var = ht.init.ones(
            shape=(self.num_channels, ), name=self.name+'.weight')
        self.bias_var = ht.init.zeros(
            shape=(self.num_channels, ), name=self.name+'.bias')

    def __call__(self, x):
        return ht.layer_normalization_op(x, self.scale_var, self.bias_var, eps=self.eps)
