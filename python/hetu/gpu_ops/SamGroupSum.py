from __future__ import absolute_import
from .Node import Op
from .. import ndarray
from ..gpu_links import sam_group_sum_link


class SamGroupSumOp(Op):
    def __init__(self, node_A, num_local_gpus=8, ctx=None):
        super().__init__(SamGroupSumOp, [node_A], ctx)
        self.num_local_gpus = num_local_gpus

    def compute(self, input_val, output_val, stream_handle=None):
        if self.on_cpu:
            raise NotImplementedError
        else:
            sam_group_sum_link(input_val[0], output_val, self.num_local_gpus, stream_handle)
    def gradient(self, output_grad):
        return [None]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        return (input_shapes[0][0], self.num_local_gpus)


def sam_group_sum_op(node, num_local_gpus, ctx=None):
    return SamGroupSumOp(node, num_local_gpus, ctx=ctx)
