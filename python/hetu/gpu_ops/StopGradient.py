from __future__ import absolute_import
from .Node import Op


class StopGradientOp(Op):
    def __init__(self, node, ctx=None):
        super().__init__(StopGradientOp, [node], ctx)

    def compute(self, input_vals, output_val, stream_handle=None):
        raise NotImplementedError

    def gradient(self, output_grad):
        return [None]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]


def stop_gradient_op(node, ctx=None):
    return StopGradientOp(node, ctx=ctx)
