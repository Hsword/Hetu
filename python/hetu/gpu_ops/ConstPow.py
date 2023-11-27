from __future__ import absolute_import
import numpy as np
from .Node import Op
from .._base import DNNL_LIB
from ..gpu_links import const_pow, const_pow_gradient


class ConstPowOp(Op):
    def __init__(self, node, val, ctx=None):
        super().__init__(ConstPowOp, [node], ctx)
        self.val = val

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            output_val[:] = self.val ** (input_vals[0].asnumpy())
        else:
            const_pow(input_vals[0], output_val, self.val, stream_handle)

    def gradient(self, output_grad):
        return [const_pow_gradient_op(self, output_grad, self.val, ctx=self.raw_ctx)]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]


class ConstPowGradientOp(Op):
    def __init__(self, node, grad, val, ctx=None):
        super().__init__(ConstPowGradientOp, [node, grad], ctx)
        self.val = val

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            output_val[:] = input_vals[0].asnumpy(
            ) * input_vals[1].asnumpy() * np.log(self.val)
        else:
            const_pow_gradient(
                input_vals[0], input_vals[1], output_val, self.val, stream_handle)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 2
        return input_shapes[0]


def const_pow_op(node, val, ctx=None):
    """Takes the power of const with exponents in node.

    Parameters:
    ----
    node : Node
        Input variable.
    val : Scalar Value
        The constant value to be powered.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return ConstPowOp(node, val, ctx=ctx)


def const_pow_gradient_op(input_node, grad_node, val, ctx=None):
    """Gradient node of const pow operation.

    Parameters:
    ----
    input_node : Node
        Previous output node.    
    grad_node : Node
        Previous gradient node.    
    val : Scalar Value
        The constant value to be powered.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return ConstPowGradientOp(input_node, grad_node, val, ctx=ctx)
