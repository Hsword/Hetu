from __future__ import absolute_import
import numpy as np
from .Node import Op
from .._base import DNNL_LIB
from ..cpu_links import softmax as cpu_softmax
from ..gpu_links import CuDNN_softmax
from ..gpu_links import CuDNN_softmax_gradient


def softmax_func(y):
    """Numerically stable softmax."""
    b = y - np.max(y, axis=-1, keepdims=True)
    expb = np.exp(b)
    softmax = expb / np.sum(expb, axis=-1, keepdims=True)
    return softmax


def softmax_gradient_func(y, dy):
    dx = y * (dy - (dy * y).sum(axis=-1, keepdims=True))
    return dx


class SoftmaxOp(Op):
    def __init__(self, node_A, ctx=None):
        super().__init__(SoftmaxOp, [node_A], ctx)

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            if DNNL_LIB['DnnlSoftmax']:
                cpu_softmax(input_vals[0], output_val)
            else:
                output_val[:] = softmax_func(input_vals[0].asnumpy())
        else:
            CuDNN_softmax(input_vals[0], output_val, stream_handle)

    def gradient(self, output_grad):
        # Do not directly use SoftmaxOp, use SoftmaxCrossEntropyOp instead.
        # Not allowing taking 2nd derivative of SoftmaxCrossEntropyOp.
        return [softmax_gradient_op(self, output_grad, ctx=self.raw_ctx)]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]


class SoftmaxGradientOp(Op):
    def __init__(self, node_y, grad, ctx=None):
        super().__init__(SoftmaxGradientOp, [node_y, grad], ctx)

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            output_val[:] = softmax_gradient_func(
                input_vals[0].asnumpy(), input_vals[1].asnumpy())
        else:
            CuDNN_softmax_gradient(
                input_vals[0], input_vals[1], output_val, stream_handle)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 2
        return input_shapes[0]


def softmax_op(node, ctx=None):
    """ This function computes its softmax along an axis.

    Parameters:
    ----
    node : Node
        Input variable.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return SoftmaxOp(node, ctx=ctx)


def softmax_gradient_op(node_y, grad, ctx=None):
    """ This function computes softmax gradient.

    Parameters:
    ----
    node_y: Node
        Output variable of forward softmax.
    grad: Node
        Gradient variable, dy.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return SoftmaxGradientOp(node_y, grad, ctx=ctx)
