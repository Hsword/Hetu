from __future__ import absolute_import
import numpy as np
from .Node import Op
from ..gpu_links import tensor_quantize, tensor_dequantize


# TODO: finish this op
class QuantizeOp(Op):
    def __init__(self, node, digit, ctx=None):
        super().__init__(QuantizeOp, [node], ctx)
        self.digit = digit

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            raise NotImplementedError
        else:
            tensor_quantize(input_vals[0], output_val, self.digit,
                            self.scale, self.zero_point, stream_handle)

    def gradient(self, output_grad):
        return [dequantize_op(output_grad, self.digit, self.raw_ctx)]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]


def quantize_op(node, digit, ctx=None):
    return QuantizeOp(node, digit, ctx=ctx)


class DequantizeOp(Op):
    def __init__(self, node, digit, ctx=None):
        super().__init__(DequantizeOp, [node], ctx)
        self.digit = digit

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            raise NotImplementedError
        else:
            tensor_dequantize(input_vals[0], output_val, self.digit,
                              self.scale, self.zero_point, stream_handle)

    def gradient(self, output_grad):
        return [quantize_op(output_grad, self.digit, self.raw_ctx)]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]


def dequantize_op(node, digit, ctx=None):
    return DequantizeOp(node, digit, ctx=ctx)
