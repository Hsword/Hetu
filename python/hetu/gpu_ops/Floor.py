from __future__ import absolute_import
import numpy as np
from .Node import Op
from .._base import DNNL_LIB
from ..gpu_links import floor


class FloorOp(Op):
    def __init__(self, node, ctx=None):
        super().__init__(FloorOp, [node], ctx)

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            output_val[:] = np.floor(input_vals[0].asnumpy())
        else:
            floor(input_vals[0], output_val, stream_handle)

    def gradient(self, output_grad):
        return [None]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]


def floor_op(node, ctx=None):
    """Make a new instance of FloorOp and call the instance.

    Parameters:
    ----
    node : Node
        Input node.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return FloorOp(node, ctx=ctx)
