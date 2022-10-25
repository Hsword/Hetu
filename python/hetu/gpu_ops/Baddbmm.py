from __future__ import absolute_import
import numpy as np
from .Node import Op
from .._base import DNNL_LIB
from ..gpu_links import baddbmm


class BaddbmmOp(Op):
    def __init__(self, node_A, node_B, node_C, alpha=1.0, beta=1.0, ctx=None):
        super().__init__(BaddbmmOp, [node_A, node_B, node_C], ctx)
        self.alpha = alpha
        self.beta = beta

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            output_val[:] = self.alpha * np.matmul(input_vals[1].asnumpy(
            ), input_vals[2].asnumpy()) + self.beta * input_vals[0].asnumpy()
        else:
            baddbmm(input_vals[0], input_vals[1], input_vals[2],
                    output_val, self.alpha, self.beta, stream_handle)

    def gradient(self, output_grad):
        from . import batch_matmul_op
        input_grad = self.beta * output_grad
        A_grad = self.alpha * \
            batch_matmul_op(
                output_grad, self.inputs[2], trans_A=False, trans_B=True, ctx=self.raw_ctx)
        B_grad = self.alpha * \
            batch_matmul_op(
                self.inputs[1], output_grad, trans_A=True, trans_B=False, ctx=self.raw_ctx)
        return [input_grad, A_grad, B_grad]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 3
        return input_shapes[0]


def baddbmm_op(node_A, node_B, node_C, alpha=1.0, beta=1.0, ctx=None):
    """Make a new instance of BaddbmmOp and call the instance.

    Parameters:
    ----
    node_A : Node
        Input node.
    node_B : Node
        Input node.
    node_C : Node
        Input node.
    alpha : Scalar value
    beta : Scalar value

    Returns:
    ----
    A new Node instance created by Op.

    """
    return BaddbmmOp(node_A, node_B, node_C, alpha, beta, ctx=ctx)
