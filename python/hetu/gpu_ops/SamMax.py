from __future__ import absolute_import
from .Node import Op
from .. import ndarray
from ..gpu_links import sammax_link, sammax_grad_link

class SamMaxOp(Op):
    def __init__(self, node_A, node_B, node_C, num_local_gpus=8, ctx=None):
        super().__init__(SamMaxOp, [node_A, node_B, node_C], ctx)
        self.num_local_gpus = num_local_gpus

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            raise NotImplementedError
        else:
            sammax_link(input_vals[0], input_vals[1], input_vals[2], output_val, self.num_local_gpus, stream_handle)
    def gradient(self, output_grad):
        return [sammax_grad_op(output_grad, self.inputs[0], self.inputs[1], self.inputs[2], self.num_local_gpus, ctx=self.raw_ctx), None, None]

    def infer_shape(self, input_shapes):
        return input_shapes[0]


class SamMaxGradOp(Op):
    def __init__(self, output_grad, node_A, node_B, node_C, num_local_gpus=8, ctx=None):
        super().__init__(SamMaxGradOp, [output_grad, node_A, node_B, node_C], ctx)
        self.num_local_gpus = num_local_gpus

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            raise NotImplementedError
        else:
            sammax_grad_link(input_vals[0], input_vals[1], input_vals[2],input_vals[3], output_val, self.num_local_gpus, stream_handle)
    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        return input_shapes[0]



def sam_max_op(node_A, node_B, node_C, num_local_gpus, ctx=None):
    return SamMaxOp(node_A, node_B, node_C, num_local_gpus, ctx=ctx)


def sammax_grad_op(node_A, node_B, node_C, node_D,  num_local_gpus, ctx=None):
    return SamMaxGradOp(node_A, node_B, node_C, node_D, num_local_gpus, ctx=ctx)


