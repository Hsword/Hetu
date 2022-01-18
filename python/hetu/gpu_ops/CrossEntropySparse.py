from __future__ import absolute_import
import numpy as np
from .Node import Op
from .._base import DNNL_LIB
from ..gpu_links import cross_entropy_sparse
from ..gpu_links import cross_entropy_sparse_gradient


class CrossEntropySparseOp(Op):
    def __init__(self, node_y, node_y_, ignored_index, ctx=None):
        super().__init__(CrossEntropySparseOp, [node_y, node_y_], ctx)
        self.ignored_index = ignored_index

    def compute(self, input_vals, output_val, stream_handle=None):
        y = input_vals[0]
        y_ = input_vals[1]
        cross_entropy_sparse(
            y, y_, self.ignored_index, output_val, stream_handle)

    def gradient(self, output_grad):
        grad_A = crossentropy_sparse_gradient_op(
            output_grad, self.inputs[0], self.inputs[1], self.ignored_index, ctx=self.raw_ctx)
        return [grad_A, None]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 2
        assert len(input_shapes[0]) >= 2
        assert input_shapes[0][:-1] == input_shapes[1]
        return input_shapes[0][:-1]


class CrossEntropySparseGradientOp(Op):
    def __init__(self, node_grad, node_y, node_y_, ignored_index, ctx=None):
        super().__init__(CrossEntropySparseGradientOp,
                         [node_grad, node_y, node_y_], ctx)
        self.ignored_index = ignored_index

    def compute(self, input_vals, output_val, stream_handle=None):
        cross_entropy_sparse_gradient(
            input_vals[0], input_vals[1], input_vals[2], self.ignored_index, output_val, stream_handle)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 3
        return input_shapes[1]


def crossentropy_sparse_op(node_y, node_y_, ignored_index=-1, ctx=None):
    return CrossEntropySparseOp(node_y, node_y_, ignored_index, ctx=ctx)


def crossentropy_sparse_gradient_op(node_grad, node_y, node_y_, ignored_index, ctx=None):
    return CrossEntropySparseGradientOp(node_grad, node_y, node_y_, ignored_index, ctx=ctx)
