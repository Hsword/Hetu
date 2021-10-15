from __future__ import absolute_import
import numpy as np
from .Node import Op
from .._base import DNNL_LIB
from ..gpu_links import matrix_multiply
from ..cpu_links import matrix_multiply as cpu_matrix_multiply


class MatMulOp(Op):
    def __init__(self, node_A, node_B, trans_A=False, trans_B=False, ctx=None):
        super().__init__(MatMulOp, [node_A, node_B], ctx)
        self.matmul_attr_trans_A = trans_A
        self.matmul_attr_trans_B = trans_B

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            if DNNL_LIB['DnnlMatrixMultiply']:
                cpu_matrix_multiply(
                    input_vals[0], self.matmul_attr_trans_A,
                    input_vals[1], self.matmul_attr_trans_B,
                    output_val)
            else:
                input_vals = [n.asnumpy() for n in input_vals]
                if ((self.matmul_attr_trans_A is False) and
                        (self.matmul_attr_trans_B is False)):
                    output_val[:] = np.matmul(input_vals[0], input_vals[1])
                elif ((self.matmul_attr_trans_A is True) and
                        (self.matmul_attr_trans_B is False)):
                    output_val[:] = np.matmul(
                        np.transpose(input_vals[0]), input_vals[1])
                elif ((self.matmul_attr_trans_A is False) and
                        (self.matmul_attr_trans_B is True)):
                    output_val[:] = np.matmul(
                        input_vals[0], np.transpose(input_vals[1]))
                elif ((self.matmul_attr_trans_A is True) and
                        (self.matmul_attr_trans_B is True)):
                    output_val[:] = np.matmul(
                        np.transpose(input_vals[0]), np.transpose(input_vals[1]))
        else:
            matrix_multiply(
                input_vals[0], self.matmul_attr_trans_A,
                input_vals[1], self.matmul_attr_trans_B,
                output_val, stream_handle)

    def gradient(self, output_grad):
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
        return [lhs_grad, rhs_grad]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 2
        A = input_shapes[0]
        B = input_shapes[1]
        assert A[1-self.matmul_attr_trans_A] == B[self.matmul_attr_trans_B]
        return (A[self.matmul_attr_trans_A], B[1-self.matmul_attr_trans_B])

    def forward_deduce_states(self, input_statuses, status, deduce_order):
        assert len(input_statuses) == 2
        l2res_map = [
            {-1: 1, 0: 0, 1: -1},  # no trans
            {-1: 1, 1: 0, 0: -1},  # trans A
        ][self.matmul_attr_trans_A]
        r2res_map = [
            {-1: 0, 0: -1, 1: 1},  # no trans
            {-1: 0, 0: 1, 1: -1},  # trans B
        ][self.matmul_attr_trans_B]
        if deduce_order:
            if input_statuses[0].valid_all():
                order = input_statuses[0].order
                status.set_order(tuple(l2res_map[x] for x in order))
            elif input_statuses[1].valid_all():
                order = input_statuses[1].order
                status.set_order(tuple(r2res_map[x] for x in order))
        else:
            if input_statuses[0].valid_state():
                state, duplicate = input_statuses[0].get()
                res_state = (
                    state.get(int(self.matmul_attr_trans_A), 1), duplicate)
                res_duplicate = state.get(1-self.matmul_attr_trans_A, 1)
                status.set_state(res_state, res_duplicate)
            elif input_statuses[1].valid_state():
                state, duplicate = input_statuses[1].get()
                res_state = (duplicate, state.get(
                    1-self.matmul_attr_trans_B, 1))
                res_duplicate = state.get(int(self.matmul_attr_trans_B), 1)
                status.set_state(res_state, res_duplicate)

    def backward_deduce_states(self, status, input_statuses, deduce_order):
        def revert(x, whether=True):
            return (x[1], x[0]) if whether else x
        assert len(input_statuses) == 2
        res2l_map = [
            {-1: 1, 0: 0, 1: -1},  # no trans
            {-1: 0, 0: 1, 1: -1},  # trans A
        ][self.matmul_attr_trans_A]
        res2r_map = [
            {-1: 0, 0: -1, 1: 1},  # no trans
            {-1: 1, 0: -1, 1: 0},  # trans B
        ][self.matmul_attr_trans_B]
        if deduce_order:
            if status.valid_all():
                res_order = tuple(res2l_map[x] for x in status.order)
                input_statuses[0].set_order(res_order)
                res_order = tuple(res2r_map[x] for x in status.order)
                input_statuses[1].set_order(res_order)
        else:
            if status.valid_state():
                state, duplicate = status.get()
                res_state = revert(
                    (state.get(0, 1), duplicate), self.matmul_attr_trans_A)
                res_duplicate = state.get(1, 1)
                input_statuses[0].set_state(res_state, res_duplicate)
                res_state = revert(
                    (duplicate, state.get(1, 1)), self.matmul_attr_trans_B)
                res_duplicate = state.get(0, 1)
                input_statuses[1].set_state(res_state, res_duplicate)
            else:
                if input_statuses[0].state is not None:
                    input_statuses[1].set_state(
                        None, input_statuses[0].state.get(int(self.matmul_attr_trans_A), 1))
                if input_statuses[1].state is not None:
                    input_statuses[0].set_state(
                        None, input_statuses[1].state.get(1 - self.matmul_attr_trans_B, 1))


def matmul_op(node_A, node_B, trans_A=False, trans_B=False, ctx=None):
    """Make a new instance of Matrix Multiplication and call the instance.

    Parameters:
    ----
    node_A : Node
        The left operand of the matrix multiplication.
    node_B : Node
        The right operand of the matrix multiplication.
    trans_A : Boolean
        Whether node_A to be transposed
    trans_B : Boolean
        Whether node_B to be transposed

    Returns:
    ----
    A new Node instance created by Op.

    """
    return MatMulOp(node_A, node_B, trans_A, trans_B, ctx=ctx)
