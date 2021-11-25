from __future__ import absolute_import
import numpy as np
from .Node import Op
from .._base import DNNL_LIB
from ..cpu_links import gelu as cpu_gelu
from ..cpu_links import gelu_gradient as cpu_gelu_gradient
from ..gpu_links import gelu
from ..gpu_links import gelu_gradient


class GeluOp(Op):
    def __init__(self, node_A, ctx=None):
        super().__init__(GeluOp, [node_A], ctx)

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            if DNNL_LIB['DnnlGelu']:
                cpu_gelu(input_vals[0], output_val)
            else:
                output_val[:] = np.maximum(input_vals[0].asnumpy(), 0)
        else:
            gelu(input_vals[0], output_val, stream_handle)
    def gradient(self, output_grad):
        return [gelu_gradient_op(self.inputs[0], output_grad, ctx=self.raw_ctx)]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]

    def get_default_state(self, status, enforce_order):
        if enforce_order:
            super().get_default_state(status, enforce_order)
        else:
            status.set_state(None, 1)


class GeluGradientOp(Op):
    def __init__(self, node_A, node_B, ctx=None):
        super().__init__(GeluGradientOp, [node_A, node_B], ctx)

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            if DNNL_LIB['DnnlGelu_Gradient']:
                cpu_gelu_gradient(input_vals[0], input_vals[1], output_val)
            else:
                output_val[:] = (np.sign(input_vals[0].asnumpy()) +
                                 1) * 0.5 * input_vals[1].asnumpy()
        else:
            gelu_gradient(input_vals[0], input_vals[1],
                          output_val, stream_handle)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 2
        return input_shapes[0]

    def get_default_state(self, status, enforce_order):
        if enforce_order:
            super().get_default_state(status, enforce_order)
        else:
            status.set_state(None, 1)


def gelu_op(node, ctx=None):
    """Rectified Linear Unit.

    Parameters:
    ----
    node : Node
        Input variable.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return GeluOp(node, ctx=ctx)


def gelu_gradient_op(node_A, node_B, ctx=None):
    """Computes the gradient of the GeLU function.  

    Parameters:
    ----
    node_A : Node
        Gselu input.
    node_B : Node
        Previous gradient node.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return GeluGradientOp(node_A, node_B, ctx=ctx)
