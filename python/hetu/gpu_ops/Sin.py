from __future__ import absolute_import
import numpy as np
from .Node import Op
from .._base import DNNL_LIB
from ..gpu_links import sin, cos


class SinOp(Op):
    def __init__(self, node, ctx=None):
        super().__init__(SinOp, [node], ctx)

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            output_val[:] = np.sin(input_vals[0].asnumpy())
        else:
            sin(input_vals[0], output_val, stream_handle)

    def gradient(self, output_grad):
        return [cos_op(self.inputs[0])*output_grad]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]


class CosOp(Op):
    def __init__(self, node, ctx=None):
        super().__init__(CosOp, [node], ctx)

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            output_val[:] = np.cos(input_vals[0].asnumpy())
        else:
            cos(input_vals[0], output_val, stream_handle)

    def gradient(self, output_grad):
        return [(-1)*sin_op(self.inputs[0])*output_grad]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]


def sin_op(node, ctx=None):
    """Returns a new node with the sine of the elements of input.

    Parameters:
    ----
    node : Node
        Input variable.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return SinOp(node, ctx=ctx)


def cos_op(node, ctx=None):
    """Returns a new node with the cosine  of the elements of input.

    Parameters:
    ----
    node : Node
        Input variable.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return CosOp(node, ctx=ctx)
