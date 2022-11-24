from __future__ import absolute_import
import numpy as np
from .Node import Op
from .._base import DNNL_LIB
from ..gpu_links import flip


class FlipOp(Op):
    def __init__(self, node_A, dims, ctx=None):
        super().__init__(FlipOp, [node_A], ctx)
        self.dims = dims

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            output_val[:] = np.flip(input_vals[0].asnumpy(), self.dims)
        else:
            flip(input_vals[0], output_val, self.dims, stream_handle)

    def gradient(self, output_grad):
        return [flip_op(output_grad, self.dims)]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]


def flip_op(node_A, dims, ctx=None):
    """Make a new instance of FlipOp and call the instance.

    Parameters:
    ----
    node_A : Node
        Input node.
    dims : List
        Dims to flip.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return FlipOp(node_A, dims, ctx=ctx)
