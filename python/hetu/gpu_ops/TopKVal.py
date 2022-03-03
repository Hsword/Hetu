from __future__ import absolute_import
from .Node import Op
from .. import ndarray
from ..gpu_links import topk_val


class TopKValOp(Op):
    def __init__(self, node_A, node_B, ctx=None):
        super().__init__(TopKValOp, [node_A, node_B], ctx)

    def compute(self, input_val, output_val, stream_handle=None):
        if self.on_cpu:
            raise NotImplementedError
        else:
            topk_val(input_val[0], input_val[1], output_val, stream_handle)

    def gradient(self, output_grad):
        return [None, None]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 2
        return input_shapes[1]

def topk_val_op(nodeA, nodeB, ctx=None):
    return TopKValOp(nodeA, nodeB, ctx=ctx)
