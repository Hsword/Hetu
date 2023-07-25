from __future__ import absolute_import
import numpy as np
from .Node import Op
from .._base import DNNL_LIB
from ..gpu_links import triu, tril
from .. import ndarray


class TriuOp(Op):
    def __init__(self, node_A, diagonal=0, ctx=None):
        super().__init__(TriuOp, [node_A], ctx)
        self.diagonal = diagonal

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            output_val[:] = np.triu(input_vals[0].asnumpy(), self.diagonal)
        else:
            triu(input_vals[0], output_val, self.diagonal, stream_handle)

    def gradient(self, output_grad):
        return [triu_op(output_grad, self.diagonal, ctx=self.raw_ctx)]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]


class TrilOp(Op):
    def __init__(self, node_A, diagonal=0, ctx=None):
        super().__init__(TrilOp, [node_A], ctx)
        self.diagonal = diagonal

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            output_val[:] = np.tril(input_vals[0].asnumpy(), self.diagonal)
        else:
            tril(input_vals[0], output_val, self.diagonal, stream_handle)

    def gradient(self, output_grad):
        return [tril_op(output_grad, self.diagonal, ctx=self.raw_ctx)]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]

def triu_op(node_A, diagonal=0, ctx=None):
    """Make a new instance of torch.triu and call the instance.

    Parameters:
    ----
    node_A : Node
        Input Variable.
    diagonal: Int
        The diagonal to consider.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return TriuOp(node_A, diagonal, ctx=ctx)


def tril_op(node_A, diagonal=0, ctx=None):
    """Make a new instance of torch.tril and call the instance.

    Parameters:
    ----
    node_A : Node
        Input Variable.
    diagonal: Int
        The diagonal to consider.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return TrilOp(node_A, diagonal, ctx=ctx)