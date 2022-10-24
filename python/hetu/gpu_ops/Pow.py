from __future__ import absolute_import
import numpy as np
from .Node import Op
from .._base import DNNL_LIB
from ..gpu_links import pow_matrix, pow_gradient


class PowOp(Op):
    def __init__(self, node_A, eps, ctx=None):
        super().__init__(PowOp, [node_A], ctx)
        self.eps = eps

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            output_val[:] = pow(input_vals[0].asnumpy(), self.eps)
        else:
            pow_matrix(input_vals[0], output_val, self.eps, stream_handle)

    def gradient(self, output_grad):
        return [pow_gradient_op(self.inputs[0], output_grad, self.eps, ctx=self.raw_ctx)]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]


class PowGradientOp(Op):
    def __init__(self, node_A, node_B, eps, ctx=None):
        super().__init__(PowGradientOp, [node_A, node_B], ctx)
        self.eps = eps

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            output_val[:] = pow(input_vals[0].asnumpy(),
                                self.eps-1) * self.eps * input_vals[1].asnumpy()
        else:
            pow_gradient(input_vals[0], input_vals[1],
                         output_val, self.eps, stream_handle)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 2
        return input_shapes[0]


def pow_op(node, eps, ctx=None):
    """Pow Node.

    Parameters:
    ----
    node : Node
        Input variable.
    eps : Float

    Returns:
    ----
    A new Node instance created by Op.

    """
    return PowOp(node, eps, ctx=ctx)


def pow_gradient_op(node_A, node_B, eps, ctx=None):
    """Pow Node.

    Parameters:
    ----
    node_A : Node
        Input variable.
    node_B : Node
        Grad variable.        
    eps : Float

    Returns:
    ----
    A new Node instance created by Op.

    """
    return PowGradientOp(node_A, node_B, eps, ctx=ctx)
