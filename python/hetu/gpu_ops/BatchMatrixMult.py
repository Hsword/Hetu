from __future__ import absolute_import
import numpy as np
from .Node import Op
from ..gpu_links import batch_matrix_multiply


class BatchMatMulOp(Op):
    def __init__(self, node_A, node_B, trans_A=False, trans_B=False, ctx=None):
        super().__init__(BatchMatMulOp, [node_A, node_B], ctx)
        self.matmul_attr_trans_A = trans_A
        self.matmul_attr_trans_B = trans_B

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            ndims = len(input_vals[0])
            perm = list(range(ndims-2)) + [ndims-1, ndims-2]

            if ((self.matmul_attr_trans_A is False) and
                    (self.matmul_attr_trans_B is False)):
                output_val[:] = np.matmul(
                    input_vals[0].asnumpy(), input_vals[1].asnumpy())
            elif ((self.matmul_attr_trans_A is True) and
                    (self.matmul_attr_trans_B is False)):
                output_val[:] = np.matmul(
                    np.transpose(input_vals[0].asnumpy(), perm), input_vals[1].asnumpy())
            elif ((self.matmul_attr_trans_A is False) and
                    (self.matmul_attr_trans_B is True)):
                output_val[:] = np.matmul(
                    input_vals[0].asnumpy(), np.transpose(input_vals[1].asnumpy(), perm))
            elif ((self.matmul_attr_trans_A is True) and
                    (self.matmul_attr_trans_B is True)):
                output_val[:] = np.matmul(
                    np.transpose(input_vals[0].asnumpy(), perm), np.transpose(input_vals[1].asnumpy(), perm))
        else:
            batch_matrix_multiply(
                input_vals[0], self.matmul_attr_trans_A,
                input_vals[1], self.matmul_attr_trans_B,
                output_val, stream_handle)

    def gradient(self, output_grad):
        if ((self.matmul_attr_trans_A is False) and
                (self.matmul_attr_trans_B is False)):
            # if Y=AB, then dA=dY B^T, dB=A^T dY
            lhs_grad = batch_matmul_op(
                output_grad, self.inputs[1], trans_A=False, trans_B=True, ctx=self.raw_ctx)
            rhs_grad = batch_matmul_op(
                self.inputs[0], output_grad, trans_A=True, trans_B=False, ctx=self.raw_ctx)
        elif ((self.matmul_attr_trans_A is True) and
                (self.matmul_attr_trans_B is False)):
            # if Y=A^T B, then dA=(dY B^T)^T=B dY^T, dB=A dY
            lhs_grad = batch_matmul_op(
                self.inputs[1], output_grad, trans_A=False, trans_B=True, ctx=self.raw_ctx)
            rhs_grad = batch_matmul_op(
                self.inputs[0], output_grad, trans_A=False, trans_B=False, ctx=self.raw_ctx)
        elif ((self.matmul_attr_trans_A is False) and
                (self.matmul_attr_trans_B is True)):
            # if Y=A B^T, then dA=dY B, dB=(A^T dY)^T=dY^T A
            lhs_grad = batch_matmul_op(
                output_grad, self.inputs[1], trans_A=False, trans_B=False, ctx=self.raw_ctx)
            rhs_grad = batch_matmul_op(
                output_grad, self.inputs[0], trans_A=True, trans_B=False, ctx=self.raw_ctx)
        elif ((self.matmul_attr_trans_A is True) and
                (self.matmul_attr_trans_B is True)):
            # if Y=A^T B^T, then dA=(dY B)^T=B^T dY^T, dB=(A dY)^T=dY^T A^T
            lhs_grad = batch_matmul_op(
                self.inputs[1], output_grad, trans_A=True, trans_B=True, ctx=self.raw_ctx)
            rhs_grad = batch_matmul_op(
                output_grad, self.inputs[0], trans_A=True, trans_B=True, ctx=self.raw_ctx)
        return [lhs_grad, rhs_grad]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 2
        A = input_shapes[0]
        B = input_shapes[1]
        assert len(A) == len(B)
        assert len(A) >= 2
        for i in range(len(A)-2):
            assert A[i] == B[i]
        C = list(A)[:-2]
        shape_A = A[-2]
        shape_B = B[-1]
        k1 = A[-1]
        k2 = B[-2]
        if self.matmul_attr_trans_A == True:
            shape_A = A[-1]
            k1 = A[-2]
        if self.matmul_attr_trans_B == True:
            shape_B = B[-2]
            k2 = B[-1]
        assert k1 == k2
        C.extend([shape_A, shape_B])
        return tuple(C)


def batch_matmul_op(node_A, node_B, trans_A=False, trans_B=False, ctx=None):
    """Make a new instance of Batch Matrix Multiplication and call the instance.

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
    return BatchMatMulOp(node_A, node_B, trans_A, trans_B, ctx=ctx)
