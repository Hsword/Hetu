from __future__ import absolute_import
import numpy as np
import scipy.sparse
from .Node import Op
from .. import ndarray
from .Transpose import transpose_op
from ..gpu_links import CuSparse_Csrmv
from ..gpu_links import CuSparse_Csrmm


class CsrmvOp(Op):
    def __init__(self, node_A, node_B, trans=False, ctx=None):
        super().__init__(CsrmvOp, [node_A, node_B], ctx)
        self.csrmv_attr_trans = trans

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            assert isinstance(input_vals[0], scipy.sparse.spmatrix)
            if self.csrmv_attr_trans is False:
                output_val[:] = input_vals[0].dot(input_vals[1].asnumpy())
            else:
                output_val[:] = input_vals[0].T.dot(input_vals[1].asnumpy())
        else:
            assert isinstance(input_vals[0], ndarray.ND_Sparse_Array)
            CuSparse_Csrmv(
                input_vals[0], self.csrmv_attr_trans,
                input_vals[1], output_val, stream_handle)

    # ND_Sparse_Array gradient not implemented
    def gradient(self, output_grad):
        if self.csrmv_attr_trans is False:
            # if Y=AB, then dA=dY B^T, dB=A^T dY
            # lhs_grad = matmul_op(
            #     output_grad, self.inputs[1], trans_A=False, trans_B=True)
            rhs_grad = csrmv_op(
                self.inputs[0], output_grad, trans=True, ctx=self.raw_ctx)
        else:
            # if Y=A^T B, then dA=(dY B^T)^T=B dY^T, dB=A dY
            # lhs_grad = matmul_op(
            #     self.inputs[1], output_grad, trans_A=False, trans_B=True)
            rhs_grad = csrmv_op(
                self.inputs[0], output_grad, trans=False, ctx=self.raw_ctx)
        return [None, rhs_grad]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 2
        A = input_shapes[0]
        B = input_shapes[1]
        assert len(A) == 2 and len(B) == 1
        shape_A = A[0]
        shape_mid_1 = A[1]
        shape_mid_2 = B[0]
        if self.csrmv_attr_trans == True:
            shape_A = A[1]
            shape_mid_1 = A[0]
        assert shape_mid_1 == shape_mid_2
        return (shape_A, )


class CsrmmOp(Op):
    def __init__(self, node_A, node_B, trans_A=False, trans_B=False, ctx=None):
        super().__init__(CsrmmOp, [node_A, node_B], ctx)
        self.csrmm_attr_trans_A = trans_A
        self.csrmm_attr_trans_B = trans_B

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            assert isinstance(input_vals[0], scipy.sparse.spmatrix)
            if ((self.csrmm_attr_trans_A is False) and
                    (self.csrmm_attr_trans_B is False)):
                output_val[:] = input_vals[0].dot(input_vals[1].asnumpy())
            elif ((self.csrmm_attr_trans_A is True) and
                    (self.csrmm_attr_trans_B is False)):
                output_val[:] = input_vals[0].T.dot(input_vals[1].asnumpy())
            elif ((self.csrmm_attr_trans_A is False) and
                    (self.csrmm_attr_trans_B is True)):
                output_val[:] = input_vals[0].dot(
                    np.transpose(input_vals[1].asnumpy()))
            elif ((self.csrmm_attr_trans_A is True) and
                    (self.csrmm_attr_trans_B is True)):
                output_val[:] = input_vals[0].T.dot(
                    np.transpose(input_vals[1].asnumpy()))
        else:
            assert isinstance(input_vals[0], ndarray.ND_Sparse_Array)
            CuSparse_Csrmm(
                input_vals[0], self.csrmm_attr_trans_A,
                input_vals[1], self.csrmm_attr_trans_B,
                output_val, stream_handle)

    # ND_Sparse_Array gradient not implemented
    def gradient(self, output_grad):
        if ((self.csrmm_attr_trans_A is False) and
                (self.csrmm_attr_trans_B is False)):
            # if Y=AB, then dA=dY B^T, dB=A^T dY
            # lhs_grad = matmul_op(
            #     output_grad, self.inputs[1], trans_A=False, trans_B=True)
            # Notice: cuSparse not support left trans right not trans
            rhs_grad = csrmm_op(
                self.inputs[0], output_grad, trans_A=True, trans_B=False, ctx=self.raw_ctx)
        elif ((self.csrmm_attr_trans_A is True) and
                (self.csrmm_attr_trans_B is False)):
            # if Y=A^T B, then dA=(dY B^T)^T=B dY^T, dB=A dY
            # lhs_grad = matmul_op(
            #     self.inputs[1], output_grad, trans_A=False, trans_B=True)
            rhs_grad = csrmm_op(
                self.inputs[0], output_grad, trans_A=False, trans_B=False, ctx=self.raw_ctx)
        elif ((self.csrmm_attr_trans_A is False) and
                (self.csrmm_attr_trans_B is True)):
            # if Y=A B^T, then dA=dY B, dB=(A^T dY)^T=dY^T A
            # lhs_grad = matmul_op(
            #     output_grad, self.inputs[1], trans_A=False, trans_B=False)
            # rhs_grad = matmul_op(
            #     output_grad, self.inputs[0], trans_A=True, trans_B=False)
            # Notice: cuSparse not support left trans right not trans
            rhs_grad = transpose_op(csrmm_op(
                self.inputs[0], output_grad, trans_A=True, trans_B=False, ctx=self.raw_ctx))
        elif ((self.csrmm_attr_trans_A is True) and
                (self.csrmm_attr_trans_B is True)):
            # if Y=A^T B^T, then dA=(dY B)^T=B^T dY^T, dB=(A dY)^T=dY^T A^T
            # lhs_grad = matmul_op(
            #     self.inputs[1], output_grad, trans_A=True, trans_B=True)
            # rhs_grad = matmul_op(
            #     output_grad, self.inputs[0], trans_A=True, trans_B=True)
            rhs_grad = transpose_op(csrmm_op(
                self.inputs[0], output_grad, trans_A=False, trans_B=False, ctx=self.raw_ctx))
        # return [lhs_grad, rhs_grad]
        return [None, rhs_grad]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 2
        A = input_shapes[0]
        B = input_shapes[1]
        assert len(A) == 2 and len(B) == 2
        shape_A = A[0]
        shape_B = B[1]
        shape_mid_1 = A[1]
        shape_mid_2 = B[0]
        if self.csrmm_attr_trans_A == True:
            shape_A = A[1]
            shape_mid_1 = A[0]
        if self.csrmm_attr_trans_B == True:
            shape_B = B[0]
            shape_mid_2 = B[1]
        assert shape_mid_1 == shape_mid_2
        return (shape_A, shape_B)


def csrmv_op(node_A, node_B, trans=False, ctx=None):
    """Make a new instance of multiplication of a sparse matrix and a vector,
        and call the instance.

    Parameters:
    ----
    node_A : Node
        The left operand, a sparse matrix.
    node_B : Node
        The right operand, a vector.
    trans : Boolean
        Whether node_A to be transposed, default to be False.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return CsrmvOp(node_A, node_B, trans, ctx=ctx)


def csrmm_op(node_A, node_B, trans_A=False, trans_B=False, ctx=None):
    """Make a new instance of Sparse Matrix Multiplication and call the instance.

    Parameters:
    ----
    node_A : Node
        The left operand, a sparse matrix.
    node_B : Node
        The right operand, a dense matrix.
    trans_A : Boolean
        Whether node_A to be transposed, default to be False.
    trans_B : Boolean
        Whether node_B to be transposed, default to be False.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return CsrmmOp(node_A, node_B, trans_A, trans_B, ctx=ctx)
