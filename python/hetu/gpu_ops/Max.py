from __future__ import absolute_import
from .Node import Op
import numpy as np
from .._base import DNNL_LIB
from ..gpu_links import max, max_mat


class MaxOp(Op):
    def __init__(self, node_A, node_B=None, dim=0, keepdim=False, ctx=None):
        if (node_B):
            super().__init__(MaxOp, [node_A, node_B], ctx)
        else:
            super().__init__(MaxOp, [node_A], ctx)
        self.dim = dim
        self.keepdim = keepdim

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            if (len(input_vals) == 1):
                output_val[:] = np.max(input_vals[0].asnumpy(), self.dim)
            else:
                output_val[:] = np.maximum(
                    input_vals[0].asnumpy(), input_vals[1].asnumpy())
        else:
            if (len(input_vals) == 1):
                max(input_vals[0], output_val, self.dim, stream_handle)
            else:
                max_mat(input_vals[0], input_vals[1],
                        output_val, stream_handle)

    def gradient(self, output_grad):
        raise NotImplemented

    def infer_shape(self, input_shapes):
        if len(self.inputs) == 2:
            return input_shapes[0]

        shape = []
        for i in range(len(input_shapes[0])):
            if i != self.dim:
                shape.append(input_shapes[0][i])

        return shape


def max_op(node_A, node_B=None, dim=0, keepdim=False, ctx=None):
    """Make a new instance of MaxOp and call the instance.

    Parameters:
    ----
    node_A : Node
        Input node.
    node_B : Node
        Input node.        
    dim : Axis along which to be maximized.
    keepdim : Bool
        Whether to keep the dimension(s).

    Returns:
    ----
    A new Node instance created by Op.

    """
    return MaxOp(node_A, node_B, dim, keepdim, ctx=ctx)
