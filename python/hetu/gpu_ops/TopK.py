from __future__ import absolute_import
from .Node import Op
from .. import ndarray
from ..gpu_links import topk


class TopKOp(Op):
    def __init__(self, node_A, ctx=None):
        super().__init__(TopKOp, [node_A], ctx)

    def compute(self, input_val, output_val, output_idx, k, stream_handle=None):
        topk(input_val, output_val, output_idx, k, stream_handle)

    def gradient(self, output_grad):
        pass

    def infer_shape(self, input_shapes):
        pass

def topk_op(node, ctx=None):
    return TopKOp(node, ctx=ctx)
