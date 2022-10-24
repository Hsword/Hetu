from __future__ import absolute_import
import numpy as np
from .Node import Op
from ..gpu_links import argsort
from .. import ndarray

class ArgsortOp(Op):
    def __init__(self, node_A, dim, descending=False, ctx=None):
        super().__init__(ArgsortOp, [node_A], ctx)
        self.dim = dim
        self.descending = descending

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            inputs = input_vals[0].asnumpy()
            output_val[:] = np.argsort(inputs, axis=self.dim)
        else:
            argsort(input_vals[0], self.output, self.index, output_val, self.dim, self.descending, stream_handle)
    def gradient(self, output_grad):
        return [None]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        self.output = ndarray.empty(input_shapes[0], self.ctx)
        self.index = ndarray.empty(input_shapes[0], self.ctx)
    
        return input_shapes[0]


def argsort_op(node, dim=1, descending=False, ctx=None):
    """Returns the indices that the sorted input.

    Parameters:
    ----
    node : Node
        Input node.
    dim : Int
        Dim to sort.
    descending : bool
        The sorting order

    Returns:
    ----
    A new Node instance created by Op.

    """
    return ArgsortOp(node, dim=dim, descending=descending, ctx=ctx)
