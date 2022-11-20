from __future__ import absolute_import
from .Node import Op
from .._base import DNNL_LIB
from ..cpu_links import matrix_elementwise_add_by_const as cpu_matrix_elementwise_add_by_const
from ..gpu_links import matrix_elementwise_add_by_const


class AddByConstOp(Op):
    def __init__(self, node_A, const_val, ctx=None):
        super().__init__(AddByConstOp, [node_A], ctx)
        self.const_attr = const_val

    @property
    def desc(self):
        return self.name + '(%s, %s)' % (self.inputs[0].name, str(self.const_attr))

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            if DNNL_LIB['DnnlMatrixElementwiseAddByConst']:
                cpu_matrix_elementwise_add_by_const(
                    input_vals[0], self.const_attr, output_val)
            else:
                output_val[:] = input_vals[0].asnumpy() + self.const_attr
        else:
            matrix_elementwise_add_by_const(
                input_vals[0], self.const_attr, output_val, stream_handle)

    def gradient(self, output_grad):
        return [output_grad]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]


def addbyconst_op(node, const_val, ctx=None):
    """Make a new instance of AddByConstOp and call the instance.

    Parameters:
    ----
    node : Node
        The Node to be added.
    const_val : scalar value
        The constant value to be added.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return AddByConstOp(node, const_val, ctx=ctx)
