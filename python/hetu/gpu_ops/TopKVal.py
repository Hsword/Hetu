from __future__ import absolute_import
from .Node import Op
from .. import ndarray
from ..gpu_links import topk_val


class TopKValOp(Op):
    def __init__(self, node_A, node_B, ctx=None):
        super().__init__(TopKOp, [node_A, node_B], ctx)

    def compute(self, input_val, output_idx, output_val, k, stream_handle=None):
        topk_val(input_val, output_idx, output_val, k, stream_handle)

    def gradient(self, output_grad):
        pass

    def infer_shape(self, input_shapes):
        pass

def topk_val_op(nodeA, nodeB, ctx=None):
    return TopKValOp(nodeA, nodeB, ctx=ctx)
