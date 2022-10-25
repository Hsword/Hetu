from __future__ import absolute_import
import numpy as np
from .Node import Op
from ..gpu_links import slice_assign, slice_assign_matrix


class SliceAssignOp(Op):
    def __init__(self, node_A, begin_pos, output_shape, val, ctx=None):
        super().__init__(SliceAssignOp, [node_A], ctx)
        self.begin_pos = list(begin_pos)
        self.output_shape = list(output_shape)
        self.val = val
        assert len(self.begin_pos) == len(self.output_shape)
        for i in range(len(self.begin_pos)):
            assert self.begin_pos[i] >= 0

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            raise NotImplementedError
        else:
            slice_assign(input_vals[0], output_val, self.val,
                         self.begin_pos, self.end_pos, stream_handle)

    def gradient(self, output_grad):
        return [output_grad]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        ndim = len(input_shapes[0])
        self.end_pos = [0 for _ in range(ndim)]

        for i in range(len(input_shapes[0])):
            if self.output_shape[i] == -1:
                self.end_pos[i] = input_shapes[0][i]
            else:
                self.end_pos[i] = self.begin_pos[i] + self.output_shape[i]
            assert self.end_pos[i] <= input_shapes[0][i]
        return input_shapes[0]


class SliceAssignMatrixOp(Op):
    def __init__(self, node_A, node_B, begin_pos_A, output_shape_A, begin_pos_B, output_shape_B, ctx=None):
        super().__init__(SliceAssignMatrixOp, [node_A, node_B], ctx)
        self.begin_pos_A = list(begin_pos_A)
        self.begin_pos_B = list(begin_pos_B)
        self.output_shape_A = list(output_shape_A)
        self.output_shape_B = list(output_shape_B)
        assert len(self.begin_pos_A) == len(self.output_shape_A)
        assert len(self.begin_pos_B) == len(self.output_shape_B)

        for i in range(len(self.begin_pos_A)):
            assert self.begin_pos_A[i] >= 0
            assert self.begin_pos_B[i] >= 0

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            raise NotImplementedError
        else:
            slice_assign_matrix(input_vals[0], input_vals[1], output_val,
                                self.begin_pos_A, self.end_pos_A, self.begin_pos_B, stream_handle)

    def gradient(self, output_grad):
        return [output_grad, None]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 2
        assert len(input_shapes[0]) == len(input_shapes[1])

        ndim = len(input_shapes[0])
        self.end_pos_A = [0 for _ in range(ndim)]
        self.end_pos_B = [0 for _ in range(ndim)]
        for i in range(len(input_shapes[0])):
            if self.output_shape_A[i] == -1:
                self.end_pos_A[i] = input_shapes[0][i]
            else:
                self.end_pos_A[i] = self.begin_pos_A[i] + \
                    self.output_shape_A[i]

            if self.output_shape_B[i] == -1:
                self.end_pos_B[i] = input_shapes[1][i]
            else:
                self.end_pos_B[i] = self.begin_pos_B[i] + \
                    self.output_shape_B[i]

            assert self.end_pos_A[i] <= input_shapes[0][i]
            assert self.end_pos_B[i] <= input_shapes[1][i]
            assert self.end_pos_A[i] - \
                self.begin_pos_A[i] == self.end_pos_B[i] - self.begin_pos_B[i]

        return input_shapes[0]


def slice_assign_op(node, begin, size, val, ctx=None):
    """Slice a matrix and assign value to it.

    Parameters:
    ----
    node : Node
        The Node needed to be sliced.
    begin: tuple
        The beginning position of slice operation.
    size: tuple
        The shape(size) of output tensor.
    val: float
        The val to be assigned.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return SliceAssignOp(node, begin, size, val, ctx=ctx)


def slice_assign_matrix_op(node_A, node_B, begin_A, size_A, begin_B, size_B, ctx=None):
    """Slice a matrix and assign it with a matrix.

    Parameters:
    ----
    node_A : Node
        The Node needed to be sliced.
    node_B : Node
        The Node needed to be assigned.        
    begin_A: tuple
        The beginning position of slice operation.
    size_A: tuple
        The shape(size) of output tensor.
    begin_B: tuple
        The beginning position of assign operation.
    size_B: tuple
        The shape(size) of assign tensor.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return SliceAssignMatrixOp(node_A, node_B, begin_A, size_A, begin_B, size_B, ctx=ctx)
