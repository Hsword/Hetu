from __future__ import absolute_import
from .Node import Op
from .._base import DNNL_LIB
from ..cpu_links import opposite as cpu_opposite
from ..gpu_links import matrix_opposite


class OppositeOp(Op):
    def __init__(self, node_A, ctx=None):
        super().__init__(OppositeOp, [node_A], ctx)
        self.dtype = node_A.dtype

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            if DNNL_LIB['DnnlOpposite']:
                cpu_opposite(input_vals[0], output_val)
            else:
                output_val[:] = -input_vals[0].asnumpy()
        else:
            matrix_opposite(input_vals[0], output_val, stream_handle)

    def gradient(self, output_grad):
        return [opposite_op(output_grad, ctx=self.raw_ctx)]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]


def opposite_op(node, ctx=None):
    """Calculate the opposite of a matrix elementwisely.

    Parameters:
    ----
    node : Node
        Input variable.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return OppositeOp(node, ctx=ctx)
