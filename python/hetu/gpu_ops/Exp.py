from __future__ import absolute_import
import numpy as np
from .Node import Op
from ..gpu_links import exp_func


class ExpOp(Op):
    def __init__(self, node_A, ctx=None):
        super().__init__(ExpOp, [node_A], ctx)

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            output_val[:] = np.exp(input_vals[0].asnumpy())
        else:
            exp_func(input_vals[0], output_val, stream_handle)

    def gradient(self, output_grad):
        from . import mul_op
        return [mul_op(self, output_grad, ctx=self.raw_ctx)]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]


def exp_op(node, ctx=None):
    return ExpOp(node, ctx=ctx)
