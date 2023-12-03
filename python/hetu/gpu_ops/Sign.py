from __future__ import absolute_import
import numpy as np
from .Node import Op
from ..gpu_links import sign_func


class SignOp(Op):
    def __init__(self, node, ctx=None):
        super().__init__(SignOp, [node], ctx)

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            output_val[:] = np.sign(input_vals[0].asnumpy())
        else:
            sign_func(input_vals[0], output_val, stream_handle)

    def gradient(self, output_grad):
        return [None]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]


def sign_op(node, ctx=None):
    return SignOp(node, ctx=ctx)
