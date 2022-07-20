from __future__ import absolute_import
import numpy as np
from .Node import Op
from .._base import DNNL_LIB
from ..cpu_links import sigmoid as cpu_sigmoid
from ..gpu_links import sigmoid


class SigmoidOp(Op):
    def __init__(self, node_A, ctx=None):
        super().__init__(SigmoidOp, [node_A], ctx)

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            if DNNL_LIB['DnnlSigmoid']:
                cpu_sigmoid(input_vals[0], output_val)
            else:
                output_val[:] = 1.0/(1.0+1.0/np.exp(input_vals[0].asnumpy()))
        else:
            sigmoid(input_vals[0], output_val, stream_handle)

    def gradient(self, output_grad):
        from .MultiplyElewise import mul_op
        from .AddConst import addbyconst_op
        from .Opposite import opposite_op
        grad_A = mul_op(self, addbyconst_op(opposite_op(
            self, ctx=self.raw_ctx), 1, ctx=self.raw_ctx), ctx=self.raw_ctx)
        return [mul_op(grad_A, output_grad, ctx=self.raw_ctx)]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]


def sigmoid_op(node, ctx=None):
    """Calculate sigmoid of a matrix elementwisely.

    Parameters:
    ----
    node : Node
        Input variable.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return SigmoidOp(node, ctx=ctx)
