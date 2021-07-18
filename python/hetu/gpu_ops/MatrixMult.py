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
        shape_A = A[0]
        shape_B = B[1]
        if self.matmul_attr_trans_A == True:
            shape_A = A[1]
        if self.matmul_attr_trans_B == True:
            shape_B = B[0]
        return (shape_A, shape_B)

    def deduce_states(self, states, duplicates):
        def revert(x):
            return (x[1], x[0])

        def gcd(x, y):
            return y if x % y == 0 else gcd(y, x % y)
        if states[0] is None and states[1] is None:
            return None, min(duplicates)
        if states[0] is None:
            states[0] = (1, 1)
        if states[1] is None:
            states[1] = (1, 1)
        assert len(states[0]) == 2 and len(states[1]) == 2
        assert np.prod(states[0]) * \
            duplicates[0] == np.prod(states[1]) * duplicates[1]
        if self.matmul_attr_trans_A:
            states[0] = revert(states[0])
        if self.matmul_attr_trans_B:
            states[1] = revert(states[1])
        assert states[0][1] == states[1][0], 'Partition number of left matrix column shoule match that of right matrix row.'
        return (states[0][0], states[1][1]), gcd(max(duplicates), min(duplicates)) * states[0][1]


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
