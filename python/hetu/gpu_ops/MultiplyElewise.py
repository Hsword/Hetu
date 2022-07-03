from __future__ import absolute_import
from .Node import Op
from .._base import DNNL_LIB
from ..cpu_links import matrix_elementwise_multiply as\
    cpu_matrix_elementwise_multiply
from ..cpu_links import matrix_elementwise_multiply_by_const as\
    cpu_matrix_elementwise_multiply_by_const
from ..gpu_links import matrix_elementwise_multiply,\
    matrix_elementwise_multiply_by_const


class MulOp(Op):
    def __init__(self, node_A, node_B, ctx=None):
        super().__init__(MulOp, [node_A, node_B], ctx)

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            if DNNL_LIB['DnnlMatrixElementwiseMultiply'] and input_vals[0].shape == input_vals[1].shape:
                cpu_matrix_elementwise_multiply(
                    input_vals[0], input_vals[1], output_val)
            elif DNNL_LIB['DnnlMatrixElementwiseMultiplyByConst'] and (input_vals[0].shape == (1,) or input_vals[1].shape == (1,)):
                if input_vals[1].shape == (1,):
                    const_val = input_vals[1].asnumpy()[0]
                    cpu_matrix_elementwise_multiply_by_const(
                        input_vals[0], const_val, output_val)
                elif input_vals[0].shape == (1,):
                    const_val = input_vals[0].asnumpy()[0]
                    cpu_matrix_elementwise_multiply_by_const(
                        input_vals[1], const_val, output_val)
            else:
                output_val[:] = input_vals[0].asnumpy() * \
                    input_vals[1].asnumpy()
        else:
            if input_vals[0].shape == input_vals[1].shape:
                matrix_elementwise_multiply(
                    input_vals[0], input_vals[1], output_val, stream_handle)
            else:
                if input_vals[1].shape == (1,):
                    const_val = input_vals[1].asnumpy()[0]
                    matrix_elementwise_multiply_by_const(
                        input_vals[0], const_val, output_val, stream_handle)
                elif input_vals[0].shape == (1,):
                    const_val = input_vals[0].asnumpy()[0]
                    matrix_elementwise_multiply_by_const(
                        input_vals[1], const_val, output_val, stream_handle)

    def gradient(self, output_grad):
        return [mul_op(self.inputs[1], output_grad, ctx=self.raw_ctx),
                mul_op(self.inputs[0], output_grad, ctx=self.raw_ctx)]

    def infer_shape(self, input_shapes):
        """Need to handle input_vals[0].shape != input_vals[1].shape"""
        assert len(input_shapes) == 2
        if input_shapes[0] == input_shapes[1]:
            output = input_shapes[0]
        else:
            if input_shapes[0] == (1,):
                output = input_shapes[1]
            elif input_shapes[1] == (1,):
                output = input_shapes[0]
            else:
                assert False, "can't do elementwise multiply between variables of different sizes."
        return output

    def forward_deduce_states(self, input_statuses, status, deduce_order):
        flag = False
        for nst in input_statuses:
            if nst.enable_partial:
                flag = True
                status.copy_from(nst, deduce_order)
        if not flag:
            super().forward_deduce_states(input_statuses, status, deduce_order)

    def backward_deduce_states(self, status, input_statuses, deduce_order):
        flag = False
        for nst in input_statuses:
            if nst.enable_partial:
                flag = True
                nst.copy_from(status, deduce_order)
        if not flag:
            super().backward_deduce_states(status, input_statuses, deduce_order)


def mul_op(node_A, node_B, ctx=None):
    """Make a new instance of matrixs elementwise multiplication and call the instance.

    Parameters:
    ----
    node_a : Node
        The Node to be multiplied.
    node_b : Node
        Another Node to be multiplied.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return MulOp(node_A, node_B, ctx=ctx)
