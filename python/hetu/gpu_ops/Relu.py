from __future__ import absolute_import
import numpy as np
from .Node import Op
from .._base import DNNL_LIB
from ..cpu_links import relu as cpu_relu
from ..cpu_links import relu_gradient as cpu_relu_gradient
from ..gpu_links import relu
from ..gpu_links import relu_gradient


class ReluOp(Op):
    def __init__(self, node_A, ctx=None):
        super().__init__(ReluOp, [node_A], ctx)

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            if DNNL_LIB['DnnlRelu']:
                cpu_relu(input_vals[0], output_val)
            else:
                output_val[:] = np.maximum(input_vals[0].asnumpy(), 0)
        else:
            relu(input_vals[0], output_val, stream_handle)

    def gradient(self, output_grad):
        return [relu_gradient_op(self.inputs[0], output_grad, ctx=self.raw_ctx)]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]


class ReluGradientOp(Op):
    def __init__(self, node_A, node_B, ctx=None):
        super().__init__(ReluGradientOp, [node_A, node_B], ctx)

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            if DNNL_LIB['DnnlRelu_Gradient']:
                cpu_relu_gradient(input_vals[0], input_vals[1], output_val)
            # heaviside function, 0.5 at x=0
            else:
                output_val[:] = (np.sign(input_vals[0].asnumpy()) +
                                 1) * 0.5 * input_vals[1].asnumpy()
        else:
            relu_gradient(input_vals[0], input_vals[1],
                          output_val, stream_handle)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 2
        return input_shapes[0]


def relu_op(node, ctx=None):
    """Rectified Linear Unit.

    Parameters:
    ----
    node : Node
        Input variable.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return ReluOp(node, ctx=ctx)


def relu_gradient_op(node_A, node_B, ctx=None):
    """Computes the gradient of the ReLU function.  

    Parameters:
    ----
    node_A : Node
        Relu input.
    node_B : Node
        Previous gradient node.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return ReluGradientOp(node_A, node_B, ctx=ctx)
