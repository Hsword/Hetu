from __future__ import absolute_import
import numpy as np
from .Node import Op
from ..gpu_links import mask_func


class MaskOp(Op):
    def __init__(self, input, mask, ctx=None):
        super().__init__(MaskOp, [input, mask], ctx)
        assert mask.dtype == np.int32  # now only support int32; in future use BOOL instead!!

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            values = input_vals[0].asnumpy()
            values[input_vals[1].asnumpy() == 0] = 0
            output_val[:] = values
        else:
            mask_func(input_vals[0], input_vals[1], output_val, stream_handle)

    def gradient(self, output_grad):
        return [mask_op(output_grad, self.inputs[1], ctx=self.raw_ctx), None]

    def infer_shape(self, input_shapes):
        return input_shapes[0]


def mask_op(input, mask, ctx=None):
    return MaskOp(input, mask, ctx=ctx)
