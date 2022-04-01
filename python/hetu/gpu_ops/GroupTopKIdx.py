from __future__ import absolute_import
from .Node import Op
from .. import ndarray
from ..gpu_links import group_topk_idx


class GroupTopKIdxOp(Op):
    def __init__(self, node_A,node_B, topk=1, num_local_gpus=8, ctx=None):
        super().__init__(GroupTopKIdxOp, [node_A, node_B], ctx)
        self.k = topk
        self.num_local_gpus = num_local_gpus

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            raise NotImplementedError
        else:
            group_topk_idx(input_vals[0], input_vals[1], output_val, self.k,self.num_local_gpus, stream_handle)
    def gradient(self, output_grad):
        return [None]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 2
        return (input_shapes[0][0], self.k)


def group_topk_idx_op(node_A, node_B, topk, num_local_gpus, ctx=None):
    return GroupTopKIdxOp(node_A, node_B, topk, num_local_gpus,  ctx=ctx)
