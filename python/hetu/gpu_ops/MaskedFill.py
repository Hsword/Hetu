from __future__ import absolute_import
import numpy as np
from .Node import Op
from ..gpu_links import masked_fill


class MaskedFillOp(Op):
    def __init__(self, input, mask, val, ctx=None):
        super().__init__(MaskedFillOp, [input, mask], ctx)
        self.val = val

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            x = input_vals[0].asnumpy()
            mask = input_vals[1].asnumpy()
            x[np.where(mask == 1)] = self.val
            output_val[:] = x
        else:
            masked_fill(input_vals[0], input_vals[1],
                        output_val, self.val, stream_handle)

    def gradient(self, output_grad):
        return [None, None]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 2
        assert input_shapes[0] == input_shapes[1]
        return input_shapes[0]


def masked_fill_op(input, mask, val, ctx=None):
    """Fill a matrix with mask.

    Parameters:
    ----
    input : Node
        Input variable.
    mask : Node
        Mask val.      
    val : Float
        Val to be fill.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return MaskedFillOp(input, mask, val, ctx=ctx)
