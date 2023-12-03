from __future__ import absolute_import
from .Node import Op
from .._base import DNNL_LIB
from ..cpu_links import matrix_elementwise_multiply_by_const as cpu_matrix_elementwise_multiply_by_const
from ..gpu_links import matrix_elementwise_multiply_by_const


class MulByConstOp(Op):
    def __init__(self, node_A, const_val, const_updater=None, ctx=None):
        super().__init__(MulByConstOp, [node_A], ctx)
        self.const_attr = const_val
        self.const_updater = const_updater
        self.dtype = node_A.dtype
        # only update in training
        if self.const_updater is not None:
            self.cnt = 0

    @property
    def desc(self):
        return self.name + '(%s, %s)' % (self.inputs[0].name, str(self.const_attr))

    def compute(self, input_vals, output_val, stream_handle=None, inference=False):
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
        if self.const_updater is not None and not inference:
            self.cnt += 1
            self.const_attr = self.const_updater(self.cnt)

    def gradient(self, output_grad):
        return [mul_byconst_op(output_grad, self.const_attr, self.const_updater, ctx=self.raw_ctx)]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]


def mul_byconst_op(node_A, const_val, const_updater=None, ctx=None):
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
    return MulByConstOp(node_A, const_val, const_updater=const_updater, ctx=ctx)
