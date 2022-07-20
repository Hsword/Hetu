from __future__ import absolute_import
import numpy as np
from .Node import Op
from ..gpu_links import argmax


class ArgmaxOp(Op):
    def __init__(self, node_A, dim, ctx=None):
        super().__init__(ArgmaxOp, [node_A], ctx)
        self.dim = dim
        self.dtype = np.int32

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            inputs = input_vals[0].asnumpy()
            output_val[:] = np.argmax(inputs, axis=self.dim)
        else:
            argmax(input_vals[0], output_val, self.dim, stream_handle)

    def gradient(self, output_grad):
        return [None]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        if len(input_shapes[0]) == 1:
            return (1,)
        output_shapes = []
        for dim, value in enumerate(input_shapes[0]):
            if dim == self.dim:
                continue
            output_shapes.append(value)
        return tuple(output_shapes)


def argmax_op(node, dim = 0, ctx=None):
    """Creates a node that represents argmax.

    Parameters:
    ----
    node : Node
        The input Node.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return ArgmaxOp(node, dim=dim, ctx=ctx)
