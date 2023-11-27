from __future__ import absolute_import
from .Node import Op
from .._base import DNNL_LIB
from .Opposite import opposite_op
from .MultiplyElewise import mul_op
from ..cpu_links import matrix_elementwise_divide as\
    cpu_matrix_elementwise_divide
from ..cpu_links import matrix_elementwise_divide_by_const as\
    cpu_matrix_elementwise_divide_by_const
from ..gpu_links import matrix_elementwise_divide, matrix_elementwise_divide_handle_zero
from ..gpu_links import matrix_elementwise_divide_const
import numpy as np

# TODO: it's better to implement gradient op for DivOp and DivConstOp
# since this can reduce the memory and the computation time (a little bit)
# if the gradient ops are added, please also modify the gradients function in executor file.

# TODO: it's better to implement gradient op for DivOp and DivConstOp
# since this can reduce the memory and the computation time (a little bit)
# if the gradient ops are added, please also modify the gradients function in executor file.


class DivOp(Op):
    def __init__(self, node_A, node_B, ctx=None):
        super().__init__(DivOp, [node_A, node_B], ctx)

    def compute(self, input_vals, output_val, stream_handle=None):
        assert input_vals[0].shape == input_vals[1].shape, \
            "can't do elementwise division between variables of different sizes."
        if self.on_cpu:
            if DNNL_LIB['DnnlMatrixElementwiseDivide']:
                cpu_matrix_elementwise_divide(
                    input_vals[0], input_vals[1], output_val)
            else:
                output_val[:] = input_vals[0].asnumpy() / \
                    input_vals[1].asnumpy()
        else:
            matrix_elementwise_divide(
                input_vals[0], input_vals[1], output_val, stream_handle)

    def gradient(self, output_grad):
        dividend_grad = div_const_op(1, self.inputs[1], ctx=self.raw_ctx)
        divisor_grad = opposite_op(
            div_op(self, self.inputs[1], ctx=self.raw_ctx), ctx=self.raw_ctx)
        return [mul_op(dividend_grad, output_grad, ctx=self.raw_ctx), mul_op(divisor_grad, output_grad, ctx=self.raw_ctx)]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 2
        assert input_shapes[0] == input_shapes[1], \
            "can't do elementwise division between variables of different sizes."
        output = input_shapes[0]
        return output

    def forward_deduce_states(self, input_statuses, status, deduce_order):
        assert len(input_statuses) == len(self.inputs)
        status.copy_from(input_statuses[0], deduce_order)

    def backward_deduce_states(self, status, input_statuses, deduce_order):
        assert len(input_statuses) == len(self.inputs)
        input_statuses[0].copy_from(status, deduce_order)
        if deduce_order:
            if status.valid_all():
                input_statuses[1].set_order(status.combine_order((-2, -1)))
        else:
            if status.valid_state():
                input_statuses[1].set_state(*status.combine_state((-2, -1)))


class DivHandleZeroOp(Op):
    # if zero, just return the input
    def __init__(self, node_A, node_B, ctx=None):
        super().__init__(DivHandleZeroOp, [node_A, node_B], ctx)

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            numerator = input_vals[0].asnumpy()
            denominator = input_vals[1].asnumpy()
            is_zero = (denominator == 0)
            is_not_zero = ~is_zero
            temp_output = np.empty(output_val.shape, dtype=self.dtype)
            temp_output[is_zero] = numerator[is_zero]
            temp_output[is_not_zero] = (numerator / denominator)[is_not_zero]
            output_val[:] = temp_output
        else:
            matrix_elementwise_divide_handle_zero(
                input_vals[0], input_vals[1], output_val, stream_handle)

    def gradient(self, output_grad):
        dividend_grad = div_const_op(1, self.inputs[1], ctx=self.raw_ctx)
        divisor_grad = opposite_op(
            div_op(self, self.inputs[1], ctx=self.raw_ctx), ctx=self.raw_ctx)
        return [mul_op(dividend_grad, output_grad, ctx=self.raw_ctx), mul_op(divisor_grad, output_grad, ctx=self.raw_ctx)]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 2
        assert input_shapes[0] == input_shapes[1], \
            "can't do elementwise division between variables of different sizes."
        output = input_shapes[0]
        return output


class DivConstOp(Op):
    def __init__(self, const_val, node_A, ctx=None):
        super().__init__(DivConstOp, [node_A], ctx)
        self.const_attr = const_val

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            if DNNL_LIB['DnnlMatrixElementwiseDivideByConst']:
                cpu_matrix_elementwise_divide_by_const(
                    input_vals[0], self.const_attr, output_val)
            else:
                output_val[:] = self.const_attr / input_vals[0].asnumpy()
        else:
            matrix_elementwise_divide_const(
                self.const_attr, input_vals[0], output_val, stream_handle)

    def gradient(self, output_grad):
        divisor_grad = div_op(opposite_op(
            self, ctx=self.raw_ctx), self.inputs[0], ctx=self.raw_ctx)
        return [mul_op(divisor_grad, output_grad, ctx=self.raw_ctx)]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]

    def forward_deduce_states(self, input_statuses, status, deduce_order):
        return

    def backward_deduce_states(self, status, input_statuses, deduce_order):
        assert len(input_statuses) == len(self.inputs)
        if deduce_order:
            if status.valid_all():
                input_statuses[0].set_order(status.combine_order((-2, -1)))
        else:
            if status.valid_state():
                input_statuses[0].set_state(*status.combine_state((-2, -1)))


def div_op(node_A, node_B, ctx=None):
    """Make a new instance of matrixs elementwise division and call the instance.

    Parameters:
    ----
    node_A : Node
        The Node where elements are numerators.
    node_B : Node
        Another Node where elements are denominators.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return DivOp(node_A, node_B, ctx=ctx)


def div_handle_zero_op(node_A, node_B, ctx=None):
    return DivHandleZeroOp(node_A, node_B, ctx=ctx)


def div_const_op(const_val, node_A, ctx=None):
    """Make a new instance of matrix elementwise devide a constant value and call the instance.

    Parameters:
    ----
    const_val: scalar value
        The constant value to be mutiplied.
    node_A : Node
        The Node where elements are denominators.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return DivConstOp(const_val, node_A, ctx=ctx)
