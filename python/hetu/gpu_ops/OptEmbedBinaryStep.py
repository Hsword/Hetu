from __future__ import absolute_import
from .Node import Op
from ..gpu_links import binary_step_forward, binary_step_backward
from .MultiplyElewise import mul_op
import numpy as np


class BinaryStepOp(Op):
    def __init__(self, node, ctx=None):
        super().__init__(BinaryStepOp, [node], ctx)

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            output_val[:] = (input_vals[0].asnumpy() > 0).astype(np.float32)
        else:
            binary_step_forward(input_vals[0], output_val, stream_handle)

    def gradient(self, output_grad):
        return [mul_op(output_grad, binary_step_gradient_op(self.inputs[0], ctx=self.raw_ctx), ctx=self.raw_ctx)]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]


def binary_step_op(node, ctx=None):
    return BinaryStepOp(node, ctx=ctx)


class BinaryStepGradientOp(Op):
    def __init__(self, node, ctx=None):
        super().__init__(BinaryStepGradientOp, [node], ctx)

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            abs_res = np.abs(input_vals[0].asnumpy())
            res = 2 - 4 * abs_res
            res[abs_res > 0.4] = 0.4
            res[abs_res > 1] = 0
            output_val[:] = res
        else:
            binary_step_backward(input_vals[0], output_val, stream_handle)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]


def binary_step_gradient_op(node, ctx=None):
    return BinaryStepGradientOp(node, ctx=ctx)
