from __future__ import absolute_import
import numpy as np
from .Node import Op
from .._base import DNNL_LIB
from ..gpu_links import array_set, as_strided, as_strided_gradient


class AsStridedOp(Op):
    def __init__(self, node_A, size, stride, ctx=None):
        super().__init__(AsStridedOp, [node_A], ctx)
        self.size = size
        self.stride = stride
        assert len(size) == len(stride)
        
    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            output_val[:] = np.lib.stride_tricks.as_strided(input_vals[0].asnumpy(), self.size, self.stride)
        else:
            as_strided(input_vals[0], output_val, self.stride, stream_handle)

    def gradient(self, output_grad):
        return [as_strided_gradient_op(self.inputs[0], output_grad, self.stride, ctx=self.raw_ctx)]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        return self.size


class AsStrided_GradientOp(Op):
    def __init__(self, node_A, node_B, stride, ctx=None):
        super().__init__(AsStrided_GradientOp, [node_A, node_B], ctx)
        self.stride = stride
        
    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            raise NotImplementedError
        else:
            array_set(output_val, 0)
            as_strided_gradient(input_vals[1], output_val, self.stride, stream_handle)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 2
        return input_shapes[0]


def as_strided_op(node_A, size, stride, ctx=None):
    """Make a new instance of AsStrideOp and call the instance.

    Parameters:
    ----
    node_A : Node
        Input node.
    Size : List
        Output shape.
    Stride : List
        Output stride.
    Returns:
    ----
    A new Node instance created by Op.

    """
    return AsStridedOp(node_A, size, stride, ctx=ctx)


def as_strided_gradient_op(node_A, node_B, stride, ctx=None):
    """Make a new instance of Abs_GradientOp and call the instance.

    Parameters:
    ----
    node_A : Node
        Input node.
    node_B : Node
        Grad node.        
    Stride : List
        Output stride.

    Returns:
    ----
    A new Node instance created by Op.

    """    
    return AsStrided_GradientOp(node_A, node_B, stride, ctx=ctx)