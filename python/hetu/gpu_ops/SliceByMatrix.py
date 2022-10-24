from __future__ import absolute_import
import numpy as np
from .Node import Op
from ..gpu_links import slice_by_matrix, slice_by_matrix_gradient, array_set


class SliceByMatrixOp(Op):
    def __init__(self, node_A, index1, index2, ctx=None):
        super().__init__(SliceByMatrixOp, [node_A, index1, index2], ctx)

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            raise NotImplementedError
        else:
            slice_by_matrix(input_vals[0], input_vals[1], input_vals[2], output_val, stream_handle)

    def gradient(self, output_grad):
        return [slice_by_matrix_gradient_op(self.inputs[0], output_grad, self.inputs[1], self.inputs[2], ctx=self.raw_ctx), None, None]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 3
        assert input_shapes[1][0] == input_shapes[2][0]
        return (input_shapes[1][0], input_shapes[0][2])


class SliceByMatrixGradientOp(Op):
    def __init__(self, input, grad, index1, index2, ctx=None):
        super().__init__(SliceByMatrixGradientOp, [input, grad, index1, index2], ctx)

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            raise NotImplementedError
        else:
            array_set(output_val, 0, stream_handle)
            slice_by_matrix_gradient(input_vals[1], input_vals[2], input_vals[3], output_val, stream_handle)

    def gradient(self, output_grad):
        return [output_grad]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 4
        return input_shapes[0]


def slice_by_matrix_op(node_A, index1, index2, ctx=None):
    """Slice a node by matrix.

    Parameters:
    ----
    node_A : Node
        The Node needed to be sliced.
    index1: Node
        Index node.
    index2: Node
        Index node.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return SliceByMatrixOp(node_A, index1, index2, ctx=ctx)

def slice_by_matrix_gradient_op(input, grad, index1, index2, ctx=None):
    """Gradient of slice by matrix.

    Parameters:
    ----
    input : Node
        The input node.
    grad : Node
        The grad node.        
    index1: Node
        Index node.
    index2: Node
        Index node.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return SliceByMatrixGradientOp(input, grad, index1, index2, ctx=ctx)
