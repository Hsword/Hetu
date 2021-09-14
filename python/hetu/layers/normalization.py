from .base import BaseLayer
import hetu as ht


class BatchNorm(BaseLayer):
    def __init__(self, num_channels, name='batchnorm'):
        self.num_channels = num_channels
        self.name = name

    def __call__(self, x):
        scale_var = ht.init.ones(
            shape=(1, self.num_channels, 1, 1), name=self.name+'_scale')
        bias_var = ht.init.zeros(
            shape=(1, self.num_channels, 1, 1), name=self.name+'_bias')
        return ht.batch_normalization_op(x, scale_var, bias_var)
