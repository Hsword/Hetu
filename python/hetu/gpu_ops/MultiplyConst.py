from __future__ import absolute_import
from .Node import Op
from .._base import DNNL_LIB
from ..cpu_links import matrix_elementwise_multiply_by_const as cpu_matrix_elementwise_multiply_by_const
from ..gpu_links import matrix_elementwise_multiply_by_const


class MulByConstOp(Op):
    def __init__(self, node_A, const_val, ctx=None):
        super().__init__(MulByConstOp, [node_A], ctx)
        self.const_attr = const_val
        self.desc = self.name + '(%s, %s)' % (node_A.name, str(const_val))

    def compute(self, input_vals, output_val, stream_handle=None):
        assert self.const_attr is not None
        if self.on_cpu:
            if DNNL_LIB['DnnlMatrixElementwiseMultiplyByConst']:
                cpu_matrix_elementwise_multiply_by_const(
                    input_vals[0], self.const_attr, output_val)
            else:
                output_val[:] = input_vals[0].asnumpy() * self.const_attr
        else:
            matrix_elementwise_multiply_by_const(
                input_vals[0], self.const_attr, output_val, stream_handle)

    def gradient(self, output_grad):
        return [self.const_attr * output_grad]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]


def mul_byconst_op(node_A, const_val, ctx=None):
    """Make a new instance of MulByConstOp and call the instance.

    Parameters:
    ----
    node : Node
        The Node to be multiplied.
    const_val : scalar value
        The constant value to be mutiplied.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return MulByConstOp(node_A, const_val, ctx=ctx)
