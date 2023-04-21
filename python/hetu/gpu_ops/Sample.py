from __future__ import absolute_import
import numpy as np
from scipy.stats import truncnorm
from .Node import Op
from .._base import DNNL_LIB
from ..cpu_links import uniform_init as cpu_uniform_init, \
    normal_init as cpu_normal_init, \
    truncated_normal_init as cpu_truncated_normal_init
from ..gpu_links import uniform_init, normal_init, \
    truncated_normal_init, gumbel_init, randint_init
from ..random import get_np_rand


class UniformSampleOp(Op):
    def __init__(self, shape, low, high, ctx=None):
        super().__init__(UniformSampleOp, [], ctx)
        self.shape = shape
        self.low = low
        self.high = high

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_gpu:
            uniform_init(output_val, self.low, self.high, stream_handle)
        else:
            if DNNL_LIB['cpu_UniformInit']:
                cpu_uniform_init(output_val, self.low, self.high)
            else:
                nprs = get_np_rand(1)
                output_val[:] = nprs.uniform(
                    low=self.low, high=self.high, size=output_val.shape).astype(output_val.dtype)

    def gradient(self, output_grad):
        return []

    def infer_shape(self, input_shapes):
        return self.shape


def uniform_sample_op(shape, low, high, ctx=None):
    return UniformSampleOp(shape, low, high, ctx)


class NormalSampleOp(Op):
    def __init__(self, shape, mean, stddev, ctx=None):
        super().__init__(NormalSampleOp, [], ctx)
        self.shape = shape
        self.mean = mean
        self.stddev = stddev

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_gpu:
            normal_init(output_val, self.mean, self.stddev, stream_handle)
        else:
            if DNNL_LIB['cpu_UniformInit']:
                cpu_normal_init(output_val, self.mean, self.stddev)
            else:
                nprs = get_np_rand(1)
                output_val[:] = nprs.normal(
                    loc=self.mean, scale=self.stddev, size=output_val.shape).astype(output_val.dtype)

    def gradient(self, output_grad):
        return []

    def infer_shape(self, input_shapes):
        return self.shape


def normal_sample_op(shape, mean, stddev, ctx=None):
    return NormalSampleOp(shape, mean, stddev, ctx)


class TruncatedNormalSampleOp(Op):
    def __init__(self, shape, mean, stddev, ctx=None):
        super().__init__(TruncatedNormalSampleOp, [], ctx)
        self.shape = shape
        self.mean = mean
        self.stddev = stddev

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_gpu:
            truncated_normal_init(output_val, self.mean,
                                  self.stddev, stream_handle)
        else:
            if DNNL_LIB['cpu_UniformInit']:
                cpu_truncated_normal_init(output_val, self.mean, self.stddev)
            else:
                get_np_rand(1)
                output_val[:] = truncnorm(
                    -2.0, 2.0, loc=self.mean, scale=self.stddev).rvs(output_val.shape).astype(output_val.dtype)

    def gradient(self, output_grad):
        return []

    def infer_shape(self, input_shapes):
        return self.shape


def truncated_normal_sample_op(shape, mean, stddev, ctx=None):
    return TruncatedNormalSampleOp(shape, mean, stddev, ctx)


class GumbelSampleOp(Op):
    def __init__(self, shape, ctx=None):
        super().__init__(GumbelSampleOp, [], ctx)
        self.shape = shape

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_gpu:
            gumbel_init(output_val, stream_handle)
        else:
            nprs = get_np_rand(1)
            output_val[:] = -np.log(-np.log(nprs.uniform(
                low=0, high=1, size=output_val.shape).astype(output_val.dtype)))

    def gradient(self, output_grad):
        return []

    def infer_shape(self, input_shapes):
        return self.shape


def gumbel_sample_op(shape, ctx=None):
    return GumbelSampleOp(shape, ctx)


class RandomIntSampleOp(Op):
    def __init__(self, shape, low, high, ctx=None):
        super().__init__(RandomIntSampleOp, [], ctx)
        self.shape = shape
        self.low = low
        self.high = high
        assert self.low < self.high
        self.dtype = np.int32

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_gpu:
            randint_init(output_val, self.low, self.high, stream_handle)
        else:
            nprs = get_np_rand(1)
            output_val[:] = nprs.randint(
                low=self.low, high=self.high, size=output_val.shape).astype(output_val.dtype)

    def gradient(self, output_grad):
        return []

    def infer_shape(self, input_shapes):
        return self.shape


def randint_sample_op(shape, low, high, ctx=None):
    return RandomIntSampleOp(shape, low, high, ctx)
