from __future__ import absolute_import
import numpy as np
from .Node import Op
from ..gpu_links import norm, norm_gradient


class NormOp(Op):
    def __init__(self, node, axis, p, keepdims=True, ctx=None):
        super().__init__(NormOp, [node], ctx)
        self.axis = axis
        self.p = p
        self.keepdims = keepdims

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            inputs = input_vals[0].asnumpy()
            output_val[:] = np.linalg.norm(
                inputs, self.p, axis=self.axis, keepdims=self.keepdims)
        else:
            norm(input_vals[0], output_val, self.axis, self.p, stream_handle)

    def gradient(self, output_grad):
        return [norm_gradient_op(self.inputs[0], self, output_grad, self.axis, self.p)]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        ndim = len(input_shapes[0])
        if self.axis < 0:
            self.axis += ndim
        assert self.axis >= 0 and self.axis < ndim
        output_shape = []
        for i in range(ndim):
            if i != self.axis:
                output_shape.append(input_shapes[0][i])
            else:
                if (self.keepdims):
                    output_shape.append(1)
        return output_shape


class NormGradientOp(Op):
    def __init__(self, node, node_y, grad_y, axis, p, ctx=None):
        super().__init__(NormGradientOp, [node, node_y, grad_y], ctx)
        self.axis = axis
        self.p = p

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            raise NotImplementedError
        else:
            norm_gradient(input_vals[0], input_vals[1], input_vals[2],
                          output_val, self.axis, self.p, stream_handle)

    def gradient(self, output_grad):
        return [None]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 3
        ndim = len(input_shapes[0])
        if self.axis < 0:
            self.axis += ndim
        return input_shapes[0]


def norm_op(node, axis, p=2, keepdims=True, ctx=None):
    """Frobenius norm

    Parameters:
    ----
    node : Node
        The Node to be normed.
    axis : Int
        Dim to be normed.
    p : Int
        CONSTANT
    keepdims : bool 
        Whether to keep the dimension.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return NormOp(node, axis, p=p, keepdims=keepdims, ctx=ctx)


def norm_gradient_op(node, node_y, grad_y, axis, p, ctx=None):
    """Gradient of frobenius norm

    Parameters:
    ----
    node : Node
        Input node.
    node_y : Node
        Input node.
    grad_y : Node
        Grad node.        
    axis : Int
        Dim to be normed.
    p : Int
        CONSTANT
    keepdims : bool 
        Whether to keep the dimension.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return NormGradientOp(node, node_y, grad_y, axis, p, ctx=ctx)
