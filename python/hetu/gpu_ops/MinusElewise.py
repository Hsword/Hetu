from __future__ import absolute_import
from .Node import Op
from .._base import DNNL_LIB
from ..gpu_links import matrix_elementwise_minus


class MinusOp(Op):
    def __init__(self, node_A, node_B, ctx=None):
        super().__init__(MinusOp, [node_A, node_B], ctx)

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            output_val[:] = input_vals[0].asnumpy() - \
                input_vals[1].asnumpy()
        else:
            matrix_elementwise_minus(
                input_vals[0], input_vals[1], output_val, stream_handle)

    def gradient(self, output_grad):
        from .Opposite import opposite_op
        return [output_grad, opposite_op(output_grad, ctx=self.raw_ctx)]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 2
        assert input_shapes[0] == input_shapes[1]
        return input_shapes[0]


def minus_op(node_A, node_B, ctx=None):
    """Make a new instance of matrixs elementwise subtraction and call the instance.

    Parameters:
    ----
    node_a : Node
        The Node to be subtracted.
    node_b : Node
        Another Node to subtract.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return MinusOp(node_A, node_B, ctx=ctx)
