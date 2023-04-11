from __future__ import absolute_import
import numpy as np
from .Node import Op
from ..gpu_links import CuDNN_log_softmax
from ..gpu_links import CuDNN_log_softmax_gradient


def log_softmax_func(y):
    b = y - np.max(y, axis=-1, keepdims=True)
    expb = np.exp(b)
    result = b - np.log(np.sum(expb, axis=-1, keepdims=True))
    return result


def log_softmax_gradient_func(y, dy):
    dx = dy - np.exp(y) * dy.sum(axis=-1, keepdims=True)
    return dx


class LogSoftmaxOp(Op):
    def __init__(self, node_A, ctx=None):
        super().__init__(LogSoftmaxOp, [node_A], ctx)

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            output_val[:] = log_softmax_func(input_vals[0].asnumpy())
        else:
            CuDNN_log_softmax(input_vals[0], output_val, stream_handle)

    def gradient(self, output_grad):
        return [log_softmax_gradient_op(self, output_grad, ctx=self.raw_ctx)]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]


class LogSoftmaxGradientOp(Op):
    def __init__(self, node_y, grad, ctx=None):
        super().__init__(LogSoftmaxGradientOp, [node_y, grad], ctx)

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            output_val[:] = log_softmax_gradient_func(
                input_vals[0].asnumpy(), input_vals[1].asnumpy())
        else:
            CuDNN_log_softmax_gradient(
                input_vals[0], input_vals[1], output_val, stream_handle)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 2
        return input_shapes[0]


def log_softmax_op(node, ctx=None):
    return LogSoftmaxOp(node, ctx=ctx)


def log_softmax_gradient_op(node_y, grad, ctx=None):
    return LogSoftmaxGradientOp(node_y, grad, ctx=ctx)
