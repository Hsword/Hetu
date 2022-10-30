from __future__ import absolute_import
import numpy as np
from .Node import Op
from .._base import DNNL_LIB
from ..gpu_links import nll_loss_link, nll_loss_grad_link

class NllLossOp(Op):
    def __init__(self, input, target, cols, ctx=None):
        super().__init__(NllLossOp, [input,target,], ctx)
        self.cols = cols

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            raise NotImplementedError
        else:
            nll_loss_link(input_vals[0], input_vals[1], output_val, stream_handle)

    def gradient(self, output_grad):
        return [nll_loss_grad_op(output_grad, self.inputs[1], self.cols, ctx=self.raw_ctx), None]

    def infer_shape(self, input_shapes):
        return (1,)

class NllLossGradOp(Op):
    def __init__(self, output_grad, target, cols, ctx=None):
        super().__init__(NllLossGradOp, [output_grad, target], ctx)
        self.cols = cols

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            raise NotImplementedError
        else:
            nll_loss_grad_link(input_vals[0], input_vals[1], output_val, stream_handle)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        batch_size = input_shapes[1][0]
        return (batch_size, self.cols)


def nll_loss_op(input, target, cols, ctx=None):
    return NllLossOp(input, target, cols, ctx=ctx)

def nll_loss_grad_op(output_grad, target, cols,  ctx=None):
    return NllLossGradOp(output_grad, target, cols, ctx=ctx)

