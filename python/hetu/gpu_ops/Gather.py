from __future__ import absolute_import
import numpy as np
from .Node import Op
from .._base import DNNL_LIB
from ..gpu_links import gather, gather_gradient, array_set


class GatherOp(Op):
    def __init__(self, node, dim, index, ctx=None):
        super().__init__(GatherOp, [node, index], ctx)
        self.dim = dim

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            assert False
        else:
            gather(input_vals[0], input_vals[1],
                   output_val, self.dim, stream_handle)

    def gradient(self, output_grad):
        return [gather_gradient_op(self.inputs[0], output_grad, self.dim, self.inputs[1], ctx=self.raw_ctx), None]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 2
        assert len(input_shapes[0]) == len(input_shapes[1])
        ndim = len(input_shapes[0])
        assert self.dim < ndim and self.dim >= -ndim
        if (self.dim < 0):
            self.dim += ndim
        return input_shapes[1]


class GatherGradientOp(Op):
    def __init__(self, input, grad, dim, index, ctx=None):
        super().__init__(GatherGradientOp, [input, grad, index], ctx)
        self.dim = dim

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            assert False
        else:
            array_set(output_val, 0, stream_handle)
            gather_gradient(input_vals[1], input_vals[2],
                            output_val, self.dim, stream_handle)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 3
        assert len(input_shapes[0]) == len(input_shapes[1])
        assert len(input_shapes[0]) == len(input_shapes[2])
        ndim = len(input_shapes[1])
        assert self.dim < ndim and self.dim >= -ndim
        if (self.dim < 0):
            self.dim += ndim
        return input_shapes[0]


def gather_op(node, dim, index, ctx=None):
    """Make a new instance of GatherOp and call the instance.

    Parameters:
    ----
    node : Node
        Input node.
    dim : Axis along which to be gathered.
    idx : The index of node to be gathered.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return GatherOp(node, dim, index, ctx=ctx)


def gather_gradient_op(input, grad, dim, index, ctx=None):
    """Make a new instance of GatherGradientOp and call the instance.

    Parameters:
    ----
    input : Node
        Input node.
    grad : Node
        Grad node.        
    dim : Axis along which to be gathered.
    idx : The index of node to be gathered.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return GatherGradientOp(input, grad, dim, index, ctx=ctx)
