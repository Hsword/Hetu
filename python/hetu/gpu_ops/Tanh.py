from __future__ import absolute_import
import numpy as np
from .Node import Op
from .._base import DNNL_LIB
from ..cpu_links import tanh as cpu_tanh
from ..gpu_links import tanh


class TanhOp(Op):
    def __init__(self, node_A, ctx=None):
        super().__init__(TanhOp, [node_A], ctx)

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            if DNNL_LIB['DnnlTanh']:
                cpu_tanh(input_vals[0], output_val)
            else:
                output_val[:] = np.tanh(input_vals[0].asnumpy())
        else:
            tanh(input_vals[0], output_val, stream_handle)

    def gradient(self, output_grad):
        grad_A = 1 + -1 * \
            tanh_op(self.inputs[0], ctx=self.raw_ctx) * \
            tanh_op(self.inputs[0], ctx=self.raw_ctx)
        return [grad_A*output_grad]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]


def tanh_op(node, ctx=None):
    """Calculate tanh of a matrix elementwisely.

    Parameters:
    ----
    node : Node
        Input variable.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return TanhOp(node, ctx=ctx)
