from __future__ import absolute_import
import numpy as np
import math
from .Node import Op
from .._base import DNNL_LIB
from ..gpu_links import arange


class ArangeOp(Op):
    def __init__(self, start, end, step, ctx=None):
        super().__init__(ArangeOp, [], ctx)
        self.start = start
        self.end = end
        self.step = step

    def compute(self, input_val, output_val, stream_handle=None):
        if self.on_cpu:
            output_val[:] = np.arange(self.start, self.end, self.step)
        else:
            arange(self.start, self.end, self.step,
                   output_val, stream_handle)

    def gradient(self, output_grad):
        return [None]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 0
        return [math.ceil((self.end-self.start)/self.step)]


def arange_op(start, end, step=1.0, ctx=None):
    """Make a new instance of ArangeOp and call the instance.

    Parameters:
    ----
    start : Scalar Value
    end : Scalar Value
    step: Scalar Value

    Returns:
    ----
    A new Node instance created by Op.

    """
    return ArangeOp(start, end, step, ctx=ctx)
