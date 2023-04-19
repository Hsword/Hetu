from __future__ import absolute_import
import numpy as np
from .Node import Op
from .._base import DNNL_LIB
from ..gpu_links import log_link, log_grad_link


class LogOp(Op):
    def __init__(self, input, eps=1e-7, ctx=None):
        super().__init__(LogOp, [input, ], ctx)
        self.eps = eps

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            output_val[:] = np.log(np.maximum(input_vals[0].asnumpy(), self.eps))
        else:
            log_link(input_vals[0], output_val, self.eps, stream_handle)

    def gradient(self, output_grad):
        return [log_grad_op(output_grad, self.inputs[0], self.eps, ctx=self.raw_ctx), ]

    def infer_shape(self, input_shapes):
        return input_shapes[0]


class LogGradOp(Op):
    def __init__(self, output_grad, input, eps=1e-7, ctx=None):
        super().__init__(LogGradOp, [output_grad, input], ctx)
        self.eps = eps

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            raise NotImplementedError
        else:
            log_grad_link(input_vals[0], input_vals[1],
                          output_val, self.eps, stream_handle)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        return input_shapes[0]


def log_op(input, eps=1e-7, ctx=None):
    return LogOp(input, eps, ctx=ctx)


def log_grad_op(output_grad, input, eps=1e-7, ctx=None):
    return LogGradOp(output_grad, input, eps, ctx=ctx)
