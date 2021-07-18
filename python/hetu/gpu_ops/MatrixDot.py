from __future__ import absolute_import
from .Node import Op
import numpy as np
from .MultiplyElewise import mul_op
from .ReduceSum import reduce_sum_op
from ..gpu_links import matrix_dot

# TODO: This op may have bugs and is not complete!
# Use other ops to replace it


class MatrixDotOp(Op):
    def __init__(self, node_A, node_B, axes=0, ctx=None):
        super().__init__(MatrixDotOp, [node_A, node_B], ctx)
        self.axes = axes

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            output_val[:] = np.tensordot(
                input_vals[0], input_vals[1], axes=self.axes)
        else:
            matrix_dot(input_vals[0], input_vals[1], output_val, stream_handle)

    def gradient(self, output_grad):
        return [matrix_dot_op(output_grad, self.inputs[1], axes=1, ctx=self.raw_ctx),
                reduce_sum_op(mul_op(self.inputs[0], output_grad, ctx=self.raw_ctx), axes=1, keepdims=True, ctx=self.raw_ctx)]

    def infer_shape(self, input_shapes):
        """Need to handle input_vals[0].shape != input_vals[1].shape"""
        return input_shapes[0]


def matrix_dot_op(node_A, node_B, axes=0, ctx=None):
    """Make a new instance of matrixs elementwise multiplication and call the instance.

    Parameters:
    ----
    node_a : Node
        The Node to be multiplied.
    node_b : Node
        Another Node to be multiplied.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return MatrixDotOp(node_A, node_B, ctx=ctx)
