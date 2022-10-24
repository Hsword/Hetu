from __future__ import absolute_import
import numpy as np
from .Node import Op
from .._base import DNNL_LIB
from ..cpu_links import tanh as cpu_tanh
from ..gpu_links import tanh, tanh_gradient


class TanhOp(Op):
    def __init__(self, node_A, ctx=None):
        super().__init__(TanhOp, [node_A], ctx)

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            if DNNL_LIB['DnnlTanh']:
                cpu_tanh(input_vals[0], output_val)
            else:
                output_val[:] = np.tanh(input_vals[0].asnumpy())
        else:
            tanh(input_vals[0], output_val, stream_handle)

    def gradient(self, output_grad):
        return [tanh_gradient_op(self, output_grad, ctx=self.raw_ctx)]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]


class TanhGradientOp(Op):
    def __init__(self, forward_node, output_grad, ctx=None):
        super().__init__(TanhGradientOp, [forward_node, output_grad], ctx=ctx)

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            temp = input_vals[0].asnumpy()
            output_val[:] = (1 - temp * temp) * input_vals[1].asnumpy()
        else:
            tanh_gradient(input_vals[0], input_vals[1],
                          output_val, stream_handle)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 2
        return input_shapes[0]


def tanh_op(node, ctx=None):
    """Calculate tanh of a matrix elementwisely.

    Parameters:
    ----
    node : Node
        Input variable.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return TanhOp(node, ctx=ctx)


def tanh_gradient_op(forward_node, output_grad, ctx=None):
    return TanhGradientOp(forward_node, output_grad, ctx=ctx)
