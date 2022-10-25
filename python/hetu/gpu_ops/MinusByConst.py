from __future__ import absolute_import
from .Node import Op
from .._base import DNNL_LIB
from ..gpu_links import minus_by_const


class MinusByConstOp(Op):
    def __init__(self, node, const_val, ctx=None):
        super().__init__(MinusByConstOp, [node], ctx)
        self.const_attr = const_val

    def compute(self, input_vals, output_val, stream_handle=None):
        assert self.const_attr is not None
        if self.on_cpu:
            output_val[:] = self.const_attr - input_vals[0].asnumpy()
        else:
            minus_by_const(input_vals[0], output_val, self.const_attr, stream_handle)

    def gradient(self, output_grad):
        return [(-1) * output_grad]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]


def minus_byconst_op(node, const_val, ctx=None):
    """Minus a node by const.

    Parameters:
    ----
    node : Node
        The Node to be minuend.
    const_val : scalar value
        The constant value to minus.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return MinusByConstOp(node, const_val, ctx=ctx)
