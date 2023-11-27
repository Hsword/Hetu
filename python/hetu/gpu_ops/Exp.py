from __future__ import absolute_import
import numpy as np
from .Node import Op
from .._base import DNNL_LIB
from ..gpu_links import exp


class ExpOp(Op):
    def __init__(self, node, ctx=None):
        super().__init__(ExpOp, [node], ctx)

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            output_val[:] = np.exp(input_vals[0].asnumpy())
        else:
            exp(input_vals[0], output_val, stream_handle)

    def gradient(self, output_grad):
        return [exp_op(self.inputs[0]) * output_grad]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]


def exp_op(node, ctx=None):
    """Exp Node.

    Parameters:
    ----
    node : Node
        Input variable.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return ExpOp(node, ctx=ctx)
