from __future__ import absolute_import
from .Node import Op
import ctypes
import numpy as np
from .._base import DNNL_LIB
#from ..cpu_links import dropout as cpu_dropout
#from ..cpu_links import dropout_gradient as cpu_dropout_gradient
from ..gpu_links import dropout2d_gradient
from ..gpu_links import dropout2d


class Dropout2dOp(Op):
    def __init__(self, node_in, keep_prob, ctx=None):
        super().__init__(Dropout2dOp, [node_in], ctx)
        self.seed = ctypes.c_ulonglong(0)
        self.mask = None
        self.keep_prob = keep_prob

    def compute(self, input_vals, output_val, stream_handle=None, inference=False):
        if inference:
            if self.on_cpu:
                output_val[:] = input_vals[0].asnumpy()
            else:
                input_vals[0].copyto(output_val)
        else:
            if self.on_cpu:
                raise NotImplementedError
            else:
                dropout2d(input_vals[0], 1 - self.keep_prob,
                          output_val, self.seed, stream_handle)

    def gradient(self, output_grad):
        return [dropout2d_gradient_op(output_grad, self.keep_prob, self, ctx=self.raw_ctx)]

    def infer_shape(self, input_shapes):
        return input_shapes[0]


class Dropout2d_GradientOp(Op):
    def __init__(self, node_in, keep_prob, forward_node, ctx=None):
        super().__init__(Dropout2d_GradientOp, [node_in], ctx)
        self.forward_node = forward_node
        self.keep_prob = keep_prob

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            raise NotImplementedError
        else:
            dropout2d_gradient(
                input_vals[0], 1 - self.keep_prob, output_val, self.forward_node.seed, stream_handle)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        return input_shapes[0]


def dropout2d_op(node_in, keep_prob, ctx=None):
    """Drops elements of input variable randomly.
    Parameters:
    ----
    node_in : Node
        Input variable.
    keep_prob : float
        Probability of the results to be kept.
    Returns:
    ----
    A new Node instance created by Op.
    """
    return Dropout2dOp(node_in, keep_prob, ctx=ctx)


def dropout2d_gradient_op(node_in, keep_prob, forward_node, ctx=None):
    """Gradient node of dropout2d operation.
    Parameters:
    ----
    node_in : Node
        Input variable.
    keep_prob : float
        Probability of the results to be kept.
    Returns:
    ----
    A new Node instance created by Op.
    """
    return Dropout2d_GradientOp(node_in, keep_prob, forward_node, ctx=ctx)
