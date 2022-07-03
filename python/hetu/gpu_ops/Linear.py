from __future__ import absolute_import
import numpy as np

from .Node import Op
from .._base import DNNL_LIB
from ..gpu_links import matmul_with_bias


class LinearOp(Op):
    def __init__(self, node_A, node_B, bias, trans_A=False, trans_B=False, ctx=None):
        super().__init__(LinearOp, [node_A, node_B, bias], ctx)
        self.matmul_attr_trans_A = trans_A
        self.matmul_attr_trans_B = trans_B

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            input_vals = [n.asnumpy() for n in input_vals]
            if ((self.matmul_attr_trans_A is False) and
                    (self.matmul_attr_trans_B is False)):
                output_val[:] = np.matmul(
                    input_vals[0], input_vals[1]) + input_vals[2]
            elif ((self.matmul_attr_trans_A is True) and
                    (self.matmul_attr_trans_B is False)):
                output_val[:] = np.matmul(
                    np.transpose(input_vals[0]), input_vals[1]) + input_vals[2]
            elif ((self.matmul_attr_trans_A is False) and
                    (self.matmul_attr_trans_B is True)):
                output_val[:] = np.matmul(
                    input_vals[0], np.transpose(input_vals[1])) + input_vals[2]
            elif ((self.matmul_attr_trans_A is True) and
                    (self.matmul_attr_trans_B is True)):
                output_val[:] = np.matmul(
                    np.transpose(input_vals[0]), np.transpose(input_vals[1])) + input_vals[2]
        else:
            matmul_with_bias(
                input_vals[0], self.matmul_attr_trans_A,
                input_vals[1], self.matmul_attr_trans_B, input_vals[2],
                output_val, stream_handle)

    def gradient(self, output_grad):
        from .MatrixMult import matmul_op
        from .ReduceSum import reduce_sum_op
        if ((self.matmul_attr_trans_A is False) and
                (self.matmul_attr_trans_B is False)):
            # if Y=AB, then dA=dY B^T, dB=A^T dY
            lhs_grad = matmul_op(
                output_grad, self.inputs[1], trans_A=False, trans_B=True, ctx=self.raw_ctx)
            rhs_grad = matmul_op(
                self.inputs[0], output_grad, trans_A=True, trans_B=False, ctx=self.raw_ctx)
        elif ((self.matmul_attr_trans_A is True) and
                (self.matmul_attr_trans_B is False)):
            # if Y=A^T B, then dA=(dY B^T)^T=B dY^T, dB=A dY
            lhs_grad = matmul_op(
                self.inputs[1], output_grad, trans_A=False, trans_B=True, ctx=self.raw_ctx)
            rhs_grad = matmul_op(
                self.inputs[0], output_grad, trans_A=False, trans_B=False, ctx=self.raw_ctx)
        elif ((self.matmul_attr_trans_A is False) and
                (self.matmul_attr_trans_B is True)):
            # if Y=A B^T, then dA=dY B, dB=(A^T dY)^T=dY^T A
            lhs_grad = matmul_op(
                output_grad, self.inputs[1], trans_A=False, trans_B=False, ctx=self.raw_ctx)
            rhs_grad = matmul_op(
                output_grad, self.inputs[0], trans_A=True, trans_B=False, ctx=self.raw_ctx)
        elif ((self.matmul_attr_trans_A is True) and
                (self.matmul_attr_trans_B is True)):
            # if Y=A^T B^T, then dA=(dY B)^T=B^T dY^T, dB=(A dY)^T=dY^T A^T
            lhs_grad = matmul_op(
                self.inputs[1], output_grad, trans_A=True, trans_B=True, ctx=self.raw_ctx)
            rhs_grad = matmul_op(
                output_grad, self.inputs[0], trans_A=True, trans_B=True, ctx=self.raw_ctx)
        bias_grad = reduce_sum_op(
            output_grad, [0], keepdims=False, ctx=self.raw_ctx)
        return [lhs_grad, rhs_grad, bias_grad]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 3
        assert all([len(shape) == 2 for shape in input_shapes[:2]])
        assert len(input_shapes[2]) == 1
        A = input_shapes[0]
        B = input_shapes[1]
        bias_shape = input_shapes[2]
        shape_A = A[0]
        shape_B = B[1]
        if self.matmul_attr_trans_A == True:
            shape_A = A[1]
        if self.matmul_attr_trans_B == True:
            shape_B = B[0]
        assert bias_shape == (shape_B,)
        return (shape_A, shape_B)

    def forward_deduce_states(self, input_statuses, status, deduce_order):
        assert len(input_statuses) == 3
        from .MatrixMult import matmul_forward_deduce
        matmul_forward_deduce(input_statuses, status, deduce_order,
                              self.matmul_attr_trans_A, self.matmul_attr_trans_B)

    def backward_deduce_states(self, status, input_statuses, deduce_order):
        assert len(input_statuses) == 3
        from .MatrixMult import matmul_backward_deduce
        matmul_backward_deduce(status, input_statuses, deduce_order,
                               self.matmul_attr_trans_A, self.matmul_attr_trans_B)
        if deduce_order:
            if status.valid_all():
                input_statuses[2].set_order(
                    status.combine_order(([0, -2], -1), (1, 0)))
        else:
            if status.valid_state():
                input_statuses[2].set_state(
                    *status.combine_state(([0, -2], -1), (1, 0)))

    def deduce_generated_backward_nodes_states(self, input_statuses, status, index):
        assert index is not None
        if index == -1:
            return status.remove_partial()
        else:
            from .MatrixMult import matmul_make_backward_status
            return matmul_make_backward_status(status, self.matmul_attr_trans_A, self.matmul_attr_trans_B, index)


def linear_op(node_A, node_B, bias, trans_A=False, trans_B=False, ctx=None):
    """Make a new instance of Matrix Multiplication with bias and call the instance.

    Parameters:
    ----
    node_A : Node
        The left operand of the matrix multiplication.
    node_B : Node
        The right operand of the matrix multiplication.
    bias : Node
        The bias of linear operation.
    trans_A : Boolean
        Whether node_A to be transposed
    trans_B : Boolean
        Whether node_B to be transposed

    Returns:
    ----
    A new Node instance created by Op.

    """
    return LinearOp(node_A, node_B, bias, trans_A, trans_B, ctx=ctx)
