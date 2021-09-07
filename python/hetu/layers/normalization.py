import hetu as ht


def BatchNorm(num_channels, name='batchnorm'):
    def batch_norm(x):
        scale_var = ht.init.ones(
            shape=(1, num_channels, 1, 1), name=name+'_scale')
        bias_var = ht.init.zeros(
            shape=(1, num_channels, 1, 1), name=name+'_bias')
        return ht.batch_normalization_op(x, scale_var, bias_var)
    return batch_norm
