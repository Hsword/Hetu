from __future__ import absolute_import
import numpy as np
from .Node import Op
from .._base import DNNL_LIB
from ..gpu_links import array_set


class FullOp(Op):
    def __init__(self, size, fill_value, ctx=None):
        super().__init__(FullOp, [], ctx)
        self.size = size
        self.fill_value = fill_value

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            output_val[:] = np.full(output_val.shape, self.fill_value)
        else:
            array_set(output_val, self.fill_value, stream_handle)

    def gradient(self, output_grad):
        return [None]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 0
        return self.size


class FullLikeOp(Op):
    def __init__(self, node, fill_value, ctx=None):
        super().__init__(FullLikeOp, [node], ctx)
        self.fill_value = fill_value

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            output_val[:] = np.full(output_val.shape, self.fill_value)
        else:
            array_set(output_val, self.fill_value, stream_handle)

    def gradient(self, output_grad):
        return [None]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]


def full_op(size, fill_value, ctx=None):
    """Make a new instance of FullOp and call the instance.

    Parameters:
    ----
    size : List
        Input size.
    fill_value : Scalar Value
        Fill value.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return FullOp(size, fill_value, ctx=ctx)


def full_like_op(node, fill_value, ctx=None):
    """Make a new instance of FullLikeOp and call the instance.

    Parameters:
    ----
    node : Node
        Input node.
    fill_value : Scalar Value
        Fill value.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return FullLikeOp(node, fill_value, ctx=ctx)
