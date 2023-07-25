from __future__ import absolute_import
import numpy as np
from .Node import Op
from .._base import DNNL_LIB
from ..gpu_links import bool_matrix, bool_val


class NeOp(Op):
    def __init__(self, node_A, node_B, ctx=None):
        if isinstance(node_B, Op):
            super().__init__(NeOp, [node_A, node_B], ctx)
        else:
            super().__init__(NeOp, [node_A], ctx)
            self.val = node_B

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            input_a = input_vals[0].asnumpy()
            if len(input_vals) == 2:
                input_b = input_vals[1].asnumpy()
            else:
                input_b = self.val
            output_val[:] = (input_a != input_b).astype(np.float32)
        else:
            if len(input_vals) == 2:
                bool_matrix(input_vals[0], input_vals[1],
                            output_val, 5, stream_handle)
            else:
                bool_val(input_vals[0], output_val, self.val, 5, stream_handle)

    def gradient(self, output_grad):
        return [None, None]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 2 or len(input_shapes) == 1
        if len(input_shapes) == 1:
            return input_shapes[0]
        if len(input_shapes[0]) == len(input_shapes[1]):
            for i in range(len(input_shapes[0])):
                assert input_shapes[0][i] == input_shapes[1][i]
            return input_shapes[0]
        assert len(input_shapes[0]) == 1
        assert len(input_shapes[1]) == 2
        assert input_shapes[1][1] == 1
        assert input_shapes[0][0] == input_shapes[1][0]
        return (input_shapes[0][0], input_shapes[0][0])


def ne_op(node_A, node_B, ctx=None):
    """Not equal node.

    Parameters:
    ----
    node_A : Node
        Input variable.
    node_B : Node/ Float
        Input variable.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return NeOp(node_A, node_B, ctx=ctx)
