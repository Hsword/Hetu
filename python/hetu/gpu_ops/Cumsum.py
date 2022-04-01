from __future__ import absolute_import
import numpy as np
from .Node import Op
from .._base import DNNL_LIB
from ..gpu_links import cumsum_with_bias

class CumsumOp(Op):
    def __init__(self, node, bias, dim, ctx=None):
        super().__init__(CumsumOp, [node], ctx)
        self.bias = bias
        self.dim = dim

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            raise NotImplementedError
        else:
            cumsum_with_bias(input_vals[0], output_val, self.bias, self.dim, stream_handle)

    def gradient(self, output_grad):
        return [None]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]

    def get_default_state(self, status, enforce_order):
        if enforce_order:
            super().get_default_state(status, enforce_order)
        else:
            status.set_state(None, 1)


def cumsum_with_bias_op(node, bias = -1, dim = 0, ctx=None):
    """Cumulative Sum with bias.

    Parameters:
    ----
    node : Node
        Input variable.
    bias : extra value to add.
    dim : which dimension to perform sum.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return CumsumOp(node, bias = -1, dim = 0, ctx=ctx)

