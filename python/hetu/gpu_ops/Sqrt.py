from __future__ import absolute_import
import numpy as np
from .Node import Op
from .._base import DNNL_LIB
from ..cpu_links import sqrt as cpu_sqrt
from ..cpu_links import rsqrt as cpu_rsqrt
from ..gpu_links import matrix_sqrt
from ..gpu_links import matrix_rsqrt


class SqrtOp(Op):
    def __init__(self, node_A, ctx=None):
        super().__init__(SqrtOp, [node_A], ctx)

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            if DNNL_LIB['DnnlSqrt']:
                cpu_sqrt(input_vals[0], output_val)
            else:
                output_val[:] = np.sqrt(input_vals[0].asnumpy())
        else:
            matrix_sqrt(input_vals[0], output_val, stream_handle)

    def gradient(self, output_grad):
        return [0.5 * rsqrt_op(self.inputs[0], ctx=self.raw_ctx) * output_grad]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]


class ReciprocalSqrtOp(Op):
    def __init__(self, node_A, ctx=None):
        super().__init__(ReciprocalSqrtOp, [node_A], ctx)

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            if DNNL_LIB['DnnlReciprocalSqrt']:
                cpu_rsqrt(input_vals[0], output_val)
            else:
                output_val[:] = 1 / np.sqrt(input_vals[0].asnumpy())
        else:
            matrix_rsqrt(input_vals[0], output_val, stream_handle)

    def gradient(self, output_grad):
        from .Division import div_op
        return [-0.5 * div_op(rsqrt_op(self.inputs[0], ctx=self.raw_ctx), self.inputs[0], ctx=self.raw_ctx) * output_grad]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]


def sqrt_op(node, ctx=None):
    """Calculate square root of a matrix elementwisely.

    Parameters:
    ----
    node : Node
        Input variable.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return SqrtOp(node, ctx=ctx)


def rsqrt_op(node, ctx=None):
    """Calculate the reciprocal of square root of a matrix elementwisely.

    Parameters:
    ----
    node : Node
        Input variable.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return ReciprocalSqrtOp(node, ctx=ctx)
