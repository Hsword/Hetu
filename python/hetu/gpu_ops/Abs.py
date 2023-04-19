from __future__ import absolute_import
import numpy as np
from .Node import Op
from ..gpu_links import abs_func


class AbsOp(Op):
    def __init__(self, node, ctx=None):
        super().__init__(AbsOp, [node], ctx)

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            output_val[:] = np.abs(input_vals[0].asnumpy())
        else:
            abs_func(input_vals[0], output_val, stream_handle)

    def gradient(self, output_grad):
        from .MultiplyElewise import mul_op
        from .Sign import sign_op
        return [mul_op(output_grad, sign_op(self.inputs[0], ctx=self.raw_ctx), ctx=self.raw_ctx)]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]


def abs_op(node, ctx=None):
    return AbsOp(node, ctx=ctx)
