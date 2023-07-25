from __future__ import absolute_import
import numpy as np
from .Node import Op
from .._base import DNNL_LIB
from ..gpu_links import index_select, index_select_grad, array_set

class IndexSelectOp(Op):
    def __init__(self, input, index, dim=0, ctx=None):
        super().__init__(IndexSelectOp, [input, index], ctx)
        self.dim = dim

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            raise NotImplementedError
        else:
            index_select(input_vals[0], input_vals[1], output_val, self.dim, stream_handle)

    def gradient(self, output_grad):
        return [index_select_grad_op(output_grad, self.inputs[0], self.inputs[1], self.dim, ctx=self.raw_ctx), None]

    def infer_shape(self, input_shapes):
        #Only support one dimension select
        assert len(input_shapes) == 2
        assert len(input_shapes[1]) == 1
        output_shape = []
        for i in range(len(input_shapes[0])):
            if i!=self.dim:
                output_shape.append(input_shapes[0][i])
            else:
                output_shape.append(input_shapes[1][0])
        return output_shape


class IndexSelectGradOp(Op):
    def __init__(self, grad, input, index, dim=0, ctx=None):
        super().__init__(IndexSelectGradOp, [grad, input, index], ctx)
        self.dim = dim

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            raise NotImplementedError
        else:
            array_set(output_val, 0)
            index_select_grad(input_vals[0], input_vals[2], output_val, self.dim, stream_handle)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 3
        return input_shapes[1]


def index_select_op(input, index, dim, ctx=None):
    """Indexing the input tensor along dimension dim using the entries in index.

    Parameters:
    ----
    input : Node
        Input node.
    index : Node
        Index node.
    dim : int
        Dimension to index.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return IndexSelectOp(input, index, dim, ctx=ctx)


def index_select_grad_op(grad, input, index, dim, ctx=None):
    """Return the gradient of index select node.

    Parameters:
    ----
    grad : Node
        Grad node.
    input : Node
        Input node.
    index : Node
        Index node.
    dim : int
        Dimension to index.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return IndexSelectGradOp(grad, input, index, dim, ctx=ctx)