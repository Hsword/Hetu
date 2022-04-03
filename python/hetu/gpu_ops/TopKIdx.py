from __future__ import absolute_import
from .Node import Op
from .. import ndarray
from ..gpu_links import topk_idx


class TopKIdxOp(Op):
    def __init__(self, node_A, topk=1, ctx=None):
        super().__init__(TopKIdxOp, [node_A], ctx)
        self.k = topk

    def compute(self, input_val, output_val, stream_handle=None):
        if self.on_cpu:
            raise NotImplementedError
        else:
            topk_idx(input_val[0], output_val, self.k, stream_handle)
    def gradient(self, output_grad):
        return [None]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        return (input_shapes[0][0], self.k)


def topk_idx_op(node, topk, ctx=None):
    return TopKIdxOp(node, topk, ctx=ctx)
