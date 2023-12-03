from hetu.gpu_ops import Variable
from hetu.random import get_np_rand
from hetu import cpu_links as cpu_op
from hetu import gpu_links as gpu_op
from hetu import ndarray
import numpy as np
import ctypes


class BaseInit(object):
    def __init__(self, shape):
        self.shape = tuple(shape)

    def __call__(self, node, stream=None):
        self.node = node
        node.tensor_value = ndarray.empty(
            self.shape, ctx=node.ctx, dtype=node.dtype)
        if ndarray.is_gpu_ctx(node.ctx):
            self.init_on_gpu(stream)
        else:
            self.init_on_cpu()

    def init_on_gpu(self, stream):
        raise NotImplementedError

    def init_on_cpu(self):
        raise NotImplementedError

    def init_on_ps(self, comm, nid, param_type, init_type, arg1, arg2, opt):
        # param types: Dense 0, Sparse 1, CacheSparse 2
        if param_type == 0:
            length = np.prod(self.shape)
            width = 1
        else:
            assert len(self.shape) == 2
            length = self.shape[0]
            width = self.shape[1]
        from .random import get_seed, get_seed_seqnum, step_seqnum
        step_seqnum(1)
        seed = get_seed() + get_seed_seqnum()
        comm.InitTensor(nid, ctypes.c_int(param_type), ctypes.c_int(length), ctypes.c_int(width),
                        ctypes.c_int(init_type), ctypes.c_double(arg1), ctypes.c_double(arg2), ctypes.c_ulonglong(seed), opt[0], opt[1], opt[2])


class EmptyInit(BaseInit):
    def __init__(self, shape):
        super().__init__(shape)

    def init_on_gpu(self, stream):
        pass

    def init_on_cpu(self):
        pass

    def init_on_ps(self, comm, nid, param_type, opt):
        raise NotImplementedError


class ConstantInit(BaseInit):
    def __init__(self, constant, shape):
        super().__init__(shape)
        self.constant = constant

    def init_on_gpu(self, stream):
        gpu_op.array_set(self.node.tensor_value, self.constant, stream)

    def init_on_cpu(self):
        from ._base import DNNL_LIB
        if DNNL_LIB['cpu_ArraySet']:
            cpu_op.array_set(self.node.tensor_value, self.constant)
        else:
            self.node.tensor_value[:] = np.full(
                self.shape, self.constant).astype(np.float32)

    def init_on_ps(self, comm, nid, param_type, opt):
        super().init_on_ps(comm, nid, param_type, 0, self.constant, 1.0, opt)


class ZerosInit(ConstantInit):
    def __init__(self, shape):
        super().__init__(0.0, shape)


class OnesInit(ConstantInit):
    def __init__(self, shape):
        super().__init__(1.0, shape)


class UniformInit(BaseInit):
    def __init__(self, low, high, shape):
        super().__init__(shape)
        self.low = low
        self.high = high

    def init_on_gpu(self, stream):
        gpu_op.uniform_init(self.node.tensor_value, self.low,
                            self.high, stream)

    def init_on_cpu(self):
        from ._base import DNNL_LIB
        if DNNL_LIB['cpu_UniformInit']:
            cpu_op.uniform_init(self.node.tensor_value,
                                self.low, self.high)
        else:
            nprs = get_np_rand(1)
            self.node.tensor_value[:] = nprs.uniform(
                low=self.low, high=self.high, size=self.shape).astype(np.float32)

    def init_on_ps(self, comm, nid, param_type, opt):
        super().init_on_ps(comm, nid, param_type, 1, self.low, self.high, opt)


class GeneralXavierUniformInit(UniformInit):
    def __init__(self, gain, mode, shape):
        assert mode in ('fan_in', 'fan_out',
                        'avg'), 'Mode %s not valid.' % mode
        assert gain > 0, 'Gain value %s not valid.' % str(gain)
        assert len(
            shape) >= 2, 'General xavier requires shape to be at least 2D.'
        hw_scale = 1 if len(shape) == 2 else np.prod(shape[2:])
        fan_in = hw_scale * shape[1]
        fan_out = hw_scale * shape[0]
        if mode == 'fan_in':
            factor = fan_in
        elif mode == 'fan_out':
            factor = fan_out
        else:
            factor = (fan_in + fan_out) / 2.0
        limit = np.sqrt(gain / factor)
        super().__init__(-limit, limit, shape)


class XavierUniformInit(GeneralXavierUniformInit):
    def __init__(self, shape):
        super().__init__(3.0, 'avg', shape)


class HeUniformInit(GeneralXavierUniformInit):
    def __init__(self, shape):
        super().__init__(6.0, 'fan_in', shape)


class LecunUniformInit(GeneralXavierUniformInit):
    def __init__(self, shape):
        super().__init__(3.0, 'fan_in', shape)


class NormalInit(BaseInit):
    def __init__(self, mean, stddev, shape):
        super().__init__(shape)
        self.mean = mean
        self.stddev = stddev

    def init_on_gpu(self, stream):
        gpu_op.normal_init(self.node.tensor_value, self.mean,
                           self.stddev, stream)

    def init_on_cpu(self):
        from ._base import DNNL_LIB
        if DNNL_LIB['cpu_NormalInit']:
            cpu_op.normal_init(self.node.tensor_value,
                               self.mean, self.stddev)
        else:
            nprs = get_np_rand(1)
            self.node.tensor_value[:] = nprs.normal(
                loc=self.mean, scale=self.stddev, size=self.shape).astype(np.float32)

    def init_on_ps(self, comm, nid, param_type, opt):
        super().init_on_ps(comm, nid, param_type, 2, self.mean, self.stddev, opt)


class GeneralXavierNormalInit(NormalInit):
    def __init__(self, gain, mode, shape):
        assert mode in ('fan_in', 'fan_out', 'avg'), 'Mode not allowed.'
        assert gain > 0, 'Gain value not allowed.'
        assert len(
            shape) >= 2, 'General xavier requires shape to be at least 2D.'
        hw_scale = 1 if len(shape) == 2 else np.prod(shape[2:])
        fan_in = hw_scale * shape[1]
        fan_out = hw_scale * shape[0]
        if mode == 'fan_in':
            factor = fan_in
        elif mode == 'fan_out':
            factor = fan_out
        else:
            factor = (fan_in + fan_out) / 2.0
        scale = np.sqrt(gain / factor)
        super().__init__(0, scale, shape)


class XavierNormalInit(GeneralXavierNormalInit):
    def __init__(self, shape):
        super().__init__(1.0, 'avg', shape)


class HeNormalInit(GeneralXavierNormalInit):
    def __init__(self, shape):
        super().__init__(2.0, 'fan_in', shape)


class LecunNormalInit(GeneralXavierNormalInit):
    def __init__(self, shape):
        super().__init__(1.0, 'fan_in', shape)


class TruncatedNormalInit(BaseInit):
    def __init__(self, mean, stddev, shape):
        super().__init__(shape)
        self.mean = mean
        self.stddev = stddev

    def init_on_gpu(self, stream):
        gpu_op.truncated_normal_init(
            self.node.tensor_value, self.mean, self.stddev, stream)

    def init_on_cpu(self):
        from ._base import DNNL_LIB
        if DNNL_LIB['cpu_TruncatedNormalInit']:
            cpu_op.truncated_normal_init(
                self.node.tensor_value, self.mean, self.stddev)
        else:
            get_np_rand(np.prod(self.shape))
            from scipy.stats import truncnorm
            self.node.tensor_value[:] = truncnorm(
                -2.0, 2.0, loc=self.mean, scale=self.stddev).rvs(self.shape).astype(np.float32)

    def init_on_ps(self, comm, nid, param_type, opt):
        super().init_on_ps(comm, nid, param_type, 3, self.mean, self.stddev, opt)


class ReversedTruncatedNormalInit(BaseInit):
    def __init__(self, mean, stddev, shape):
        super().__init__(shape)
        self.mean = mean
        self.stddev = stddev

    def init_on_gpu(self, stream):
        gpu_op.reversed_truncated_normal_init(
            self.node.tensor_value, self.mean, self.stddev, stream)

    def init_on_cpu(self):
        from ._base import DNNL_LIB
        if DNNL_LIB['cpu_ReversedTruncatedNormalInit']:
            cpu_op.reversed_truncated_normal_init(
                self.node.tensor_value, self.mean, self.stddev)
        else:
            raise NotImplementedError


# here we provide easy APIs

def nulls(shape, name=None, trainable=True, dtype=np.float32, ctx=None):
    if name is None:
        name = 'empty_initializer'
    init = EmptyInit(shape)
    return Variable(name=name, initializer=init, trainable=trainable, dtype=dtype, ctx=ctx)


def zeros(shape, name=None, trainable=True, dtype=np.float32, ctx=None):
    if name is None:
        name = 'zeros_initializer'
    init = ZerosInit(shape)
    return Variable(name=name, initializer=init, trainable=trainable, dtype=dtype, ctx=ctx)


def ones(shape, name=None, trainable=True, dtype=np.float32, ctx=None):
    if name is None:
        name = 'ones_initializer'
    init = OnesInit(shape)
    return Variable(name=name, initializer=init, trainable=trainable, dtype=dtype, ctx=ctx)


def constant(shape, fill_value=0.0, name=None, trainable=True, dtype=np.float32, ctx=None):
    if name is None:
        name = 'constant_initializer'
    init = ConstantInit(fill_value, shape)
    return Variable(name=name, initializer=init, trainable=trainable, dtype=dtype, ctx=ctx)


def truncated_normal(shape, mean=0.0, stddev=1.0, name=None, trainable=True, dtype=np.float32, ctx=None):
    if name is None:
        name = 'truncated_normal_initializer'
    init = TruncatedNormalInit(mean, stddev, shape)
    return Variable(name=name, initializer=init, trainable=trainable, dtype=dtype, ctx=ctx)


def reversed_truncated_normal(shape, mean=0.0, stddev=1.0, name=None, trainable=True, dtype=np.float32, ctx=None):
    if name is None:
        name = 'reversed_truncated_normal_initializer'
    init = ReversedTruncatedNormalInit(mean, stddev, shape)
    return Variable(name=name, initializer=init, trainable=trainable, dtype=dtype, ctx=ctx)


def random_normal(shape, mean=0.0, stddev=1.0, name=None, trainable=True, dtype=np.float32, ctx=None):
    if name is None:
        name = 'random_normal_initializer'
    init = NormalInit(mean, stddev, shape)
    return Variable(name=name, initializer=init, trainable=trainable, dtype=dtype, ctx=ctx)


def random_uniform(shape, minval=-1.0, maxval=1.0, name=None, trainable=True, dtype=np.float32, ctx=None):
    if name is None:
        name = 'random_uniform_initializer'
    init = UniformInit(minval, maxval, shape)
    return Variable(name=name, initializer=init, trainable=trainable, dtype=dtype, ctx=ctx)


def general_xavier_normal(shape, gain, mode, name=None, trainable=True, dtype=np.float32, ctx=None):
    if name is None:
        name = 'general_xavier_normal_initializer'
    init = GeneralXavierNormalInit(gain, mode, shape)
    return Variable(name=name, initializer=init, trainable=trainable, dtype=dtype, ctx=ctx)


def general_xavier_uniform(shape, gain, mode, name=None, trainable=True, dtype=np.float32, ctx=None):
    if name is None:
        name = 'general_xavier_uniform_initializer'
    init = GeneralXavierUniformInit(gain, mode, shape)
    return Variable(name=name, initializer=init, trainable=trainable, dtype=dtype, ctx=ctx)


def xavier_normal(shape, name=None, trainable=True, dtype=np.float32, ctx=None):
    if name is None:
        name = 'xavier_normal_initializer'
    init = XavierNormalInit(shape)
    return Variable(name=name, initializer=init, trainable=trainable, dtype=dtype, ctx=ctx)


def xavier_uniform(shape, name=None, trainable=True, dtype=np.float32, ctx=None):
    if name is None:
        name = 'xavier_uniform_initializer'
    init = XavierUniformInit(shape)
    return Variable(name=name, initializer=init, trainable=trainable, dtype=dtype, ctx=ctx)


def he_normal(shape, name=None, trainable=True, dtype=np.float32, ctx=None):
    if name is None:
        name = 'he_normal_initializer'
    init = HeNormalInit(shape)
    return Variable(name=name, initializer=init, trainable=trainable, dtype=dtype, ctx=ctx)


def he_uniform(shape, name=None, trainable=True, dtype=np.float32, ctx=None):
    if name is None:
        name = 'he_uniform_initializer'
    init = HeUniformInit(shape)
    return Variable(name=name, initializer=init, trainable=trainable, dtype=dtype, ctx=ctx)


def lecun_normal(shape, name=None, trainable=True, dtype=np.float32, ctx=None):
    if name is None:
        name = 'lecun_normal_initializer'
    init = LecunNormalInit(shape)
    return Variable(name=name, initializer=init, trainable=trainable, dtype=dtype, ctx=ctx)


def lecun_uniform(shape, name=None, trainable=True, dtype=np.float32, ctx=None):
    if name is None:
        name = 'lecun_uniform_initializer'
    init = LecunUniformInit(shape)
    return Variable(name=name, initializer=init, trainable=trainable, dtype=dtype, ctx=ctx)


# here we provide generators

def _generate(init_func, **init_kargs):
    def _generator_helper(shape, name=None, trainable=True, dtype=np.float32, ctx=None):
        return init_func(shape=shape, name=name, trainable=trainable, dtype=dtype, ctx=ctx, **init_kargs)
    return _generator_helper


def GenEmpty():
    return _generate(nulls)


def GenZeros():
    return _generate(zeros)


def GenOnes():
    return _generate(ones)


def GenConstant(fill_value=0.0):
    return _generate(constant, fill_value=fill_value)


def GenTruncatedNormal(mean=0.0, stddev=1.0):
    return _generate(truncated_normal, mean=mean, stddev=stddev)


def GenReversedTruncatedNormal(mean=0.0, stddev=1.0):
    return _generate(reversed_truncated_normal, mean=mean, stddev=stddev)


def GenNormal(mean=0.0, stddev=1.0):
    return _generate(random_normal, mean=mean, stddev=stddev)


def GenUniform(minval=-1.0, maxval=1.0):
    return _generate(random_uniform, minval=minval, maxval=maxval)


def GenGeneralXavierNormal(gain, mode):
    return _generate(general_xavier_normal, gain=gain, mode=mode)


def GenGeneralXavierUniform(gain, mode):
    return _generate(general_xavier_uniform, gain=gain, mode=mode)


def GenXavierNormal():
    return _generate(xavier_normal)


def GenXavierUniform():
    return _generate(xavier_uniform)


def GenHeNormal():
    return _generate(he_normal)


def GenHeUniform():
    return _generate(he_uniform)


def GenLecunNormal():
    return _generate(lecun_normal)


def GenLecunUniform():
    return _generate(lecun_uniform)
