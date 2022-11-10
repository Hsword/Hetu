from __future__ import absolute_import
import numpy as np
from .Node import Op
from .._base import DNNL_LIB
"""
from ..cpu_links import leaky_relu as cpu_leaky_relu
from ..cpu_links import leaky_relu_gradient as cpu_leaky_relu_gradient
"""
from ..gpu_links import leaky_relu
from ..gpu_links import leaky_relu_gradient


class LeakyReluOp(Op):
    def __init__(self, node_A, const_val, ctx=None):
        super().__init__(LeakyReluOp, [node_A], ctx)
        self.const_attr = const_val

    @property
    def desc(self):
        return self.name + '(%s, %s)' % (self.inputs[0].name, str(self.const_attr))

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            raise NotImplementedError
        else:
            leaky_relu(input_vals[0], self.const_attr,
                       output_val, stream_handle)

    def gradient(self, output_grad):
        return [leaky_relu_gradient_op(self.inputs[0], output_grad, self.const_attr, ctx=self.raw_ctx)]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]


class LeakyReluGradientOp(Op):
    def __init__(self, node_A, node_B, const_val, ctx=None):
        super().__init__(LeakyReluGradientOp, [node_A, node_B], ctx)
        self.const_attr = const_val

    @property
    def desc(self):
        return self.name + \
            '(%s, %s, %s)' % (self.inputs[0].name,
                              self.inputs[1].name, str(self.const_attr))

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            raise NotImplementedError
        else:
            leaky_relu_gradient(
                input_vals[0], input_vals[1], self.const_attr, output_val, stream_handle)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 2
        return input_shapes[0]


def leaky_relu_op(node, alpha, ctx=None):
    """Rectified Linear Unit.

    Parameters:
    ----
    node : Node
        Input variable.
    alpha : float
        LeakyRelu's alpha 

    Returns:
    ----
    A new Node instance created by Op.

    """
    return LeakyReluOp(node, alpha, ctx=ctx)


def leaky_relu_gradient_op(node_A, node_B, alpha, ctx=None):
    """Computes the gradient of the ReLU function.  

    Parameters:
    ----
    node_A : Node
        LeakyRelu input.
    node_B : Node
        Previous gradient node.
    alpha : float
        LeakyRelu alpha

    Returns:
    ----
    A new Node instance created by Op.

    """
    return LeakyReluGradientOp(node_A, node_B, alpha, ctx=ctx)
