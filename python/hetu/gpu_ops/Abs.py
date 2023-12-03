from __future__ import absolute_import
import numpy as np
from .Node import Op
from .._base import DNNL_LIB
from ..gpu_links import abs_val, abs_gradient


class AbsOp(Op):
    def __init__(self, node_A, ctx=None):
        super().__init__(AbsOp, [node_A], ctx)
        
    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            output_val[:] = np.abs(input_vals[0].asnumpy())
        else:
            abs_val(input_vals[0], output_val, stream_handle)

    def gradient(self, output_grad):
        return [abs_gradient_op(self.inputs[0], output_grad, ctx=self.raw_ctx)]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]


class Abs_GradientOp(Op):
    def __init__(self, node_A, node_B, ctx=None):
        super().__init__(Abs_GradientOp, [node_A, node_B], ctx)
        
    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            output_val[:] = np.sign(input_vals[0].asnumpy()) * input_vals[1].asnumpy()
        else:
            abs_gradient(input_vals[0], input_vals[1], output_val, stream_handle)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 2
        return input_shapes[0]


def abs_op(node_A, ctx=None):
    """Make a new instance of AbsOp and call the instance.

    Parameters:
    ----
    node_A : Node
        Input node.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return AbsOp(node_A, ctx=ctx)


def abs_gradient_op(node_A, node_B, ctx=None):
    """Make a new instance of Abs_GradientOp and call the instance.

    Parameters:
    ----
    node_A : Node
        Input node.
    node_B : Node
        Grad node.        

    Returns:
    ----
    A new Node instance created by Op.

    """    
    return Abs_GradientOp(node_A, node_B, ctx=ctx)
