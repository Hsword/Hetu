from __future__ import absolute_import
import numpy as np
from .Node import Op
from .._base import DNNL_LIB
from ..gpu_links import bool_func

class BoolOp(Op):
    def __init__(self, node, ctx=None):
        super().__init__(BoolOp, [node], ctx)

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            raise NotImplementedError
        else:
            bool_func(input_vals[0], output_val, stream_handle)

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


def bool_op(node, ctx=None):
    """Boolean Node.

    Parameters:
    ----
    node : Node
        Input variable.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return BoolOp(node, ctx=ctx)

