from __future__ import absolute_import
import numpy as np
from .Node import Op
from .._base import DNNL_LIB
from ..gpu_links import fmod
import math


class FmodOp(Op):
    def __init__(self, node, val, ctx=None):
        super().__init__(FmodOp, [node], ctx)
        self.val = val

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            input_val = input_vals[0].asnumpy()
            output_val[:] = input_val - math.trunc(input_val/self.val)*self.val
        else:
            fmod(input_vals[0], output_val, self.val, stream_handle)

    def gradient(self, output_grad):
        return [output_grad]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]


def fmod_op(node, val, ctx=None):
    """Make a new instance of FloorOp and call the instance.

    Parameters:
    ----
    node : Node
        Input node.
    val : Float
        Val to mod.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return FmodOp(node, val, ctx=ctx)
