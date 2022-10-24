from __future__ import absolute_import
import numpy as np
from .Node import Op
from .MatrixMult import matmul_op
from .._base import DNNL_LIB
from ..gpu_links import addmm, addmm_gradient, broadcast_shape_simple, matrix_elementwise_multiply_by_const
from .. import ndarray


class AddmmOp(Op):
    def __init__(self, node_A, node_B, node_C, alpha=1.0, beta=1.0, ctx=None):
        super().__init__(AddmmOp, [node_A, node_B, node_C], ctx)
        self.alpha = alpha
        self.beta = beta

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            matA = input_vals[0].asnumpy()
            matB = input_vals[1].asnumpy()
            matC = input_vals[2].asnumpy()
            output_val[:] = self.beta * matA + self.alpha * (matB@matC)
        else:
            input_a = input_vals[0]
            if len(input_vals[0].shape) == 1:
                tmp_val = ndarray.empty(
                    output_val.shape, ctx=input_vals[0].ctx)
                broadcast_shape_simple(
                    input_vals[0], tmp_val, self.out_strides, self.in_dims, stream_handle)
                input_a = tmp_val
            addmm(input_a, input_vals[1], input_vals[2],
                  output_val, self.alpha, self.beta, stream_handle)

    def gradient(self, output_grad):
        input_grad = addmm_gradient_op(
            self.inputs[0], output_grad, self.beta, ctx=self.raw_ctx)
        A_grad = self.alpha * \
            matmul_op(
                output_grad, self.inputs[2], trans_A=False, trans_B=True, ctx=self.raw_ctx)
        B_grad = self.alpha * \
            matmul_op(self.inputs[1], output_grad,
                      trans_A=True, trans_B=False, ctx=self.raw_ctx)
        return [input_grad, A_grad, B_grad]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 3
        assert len(input_shapes[1]) == 2
        assert len(input_shapes[2]) == 2
        assert input_shapes[1][1] == input_shapes[2][0]
        self.output_shape = (input_shapes[1][0], input_shapes[2][1])
        if len(input_shapes[0]) == 1:
            assert input_shapes[0][0] == input_shapes[2][1]
            out_strides = [self.output_shape[1], 1]
            in_dims = [1, self.output_shape[1]]
            self.out_strides = ndarray.array(
                out_strides, self.ctx, data_type=np.int32)
            self.in_dims = ndarray.array(in_dims, self.ctx, data_type=np.int32)
        else:
            assert len(input_shapes[0]) == 2
            assert input_shapes[0][0] == input_shapes[1][0]
            assert input_shapes[0][1] == input_shapes[2][1]
        return self.output_shape


class AddmmGradientOp(Op):
    def __init__(self, node_input, node_grad, beta=1.0, ctx=None):
        super().__init__(AddmmGradientOp, [node_input, node_grad], ctx)
        self.beta = beta
        self.reduce_sum = False

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            if self.reduce_sum:
                matA = input_vals[1].asnumpy()
                temp = np.sum(self.beta * matA, axis=0)
                output_val[:] = temp
            else:
                matA = input_vals[1].asnumpy()
                output_val[:] = self.beta * matA
        else:
            if self.reduce_sum:
                addmm_gradient(input_vals[1], output_val,
                               0, self.beta, stream_handle)
            else:
                matrix_elementwise_multiply_by_const(
                    input_vals[1], self.beta, output_val, stream_handle)

    def gradient(self, output_grad):
        raise NotImplemented

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 2
        assert len(input_shapes[1]) == 2
        if len(input_shapes[0]) == 1:
            assert input_shapes[0][0] == input_shapes[1][1]
            self.reduce_sum = True
        else:
            assert len(input_shapes[0]) == 2
            assert input_shapes[0][0] == input_shapes[1][0]
            assert input_shapes[0][1] == input_shapes[1][1]
        return input_shapes[0]


def addmm_op(node_A, node_B, node_C, alpha=1.0, beta=1.0, ctx=None):
    """Make a new instance of AddmmOp and call the instance.

    Parameters:
    ----
    node_A : Node
        Input node.
    node_B : Node
        Input node.
    node_C : Node
        Input node.
    alpha : Scalar value
    beta : Scalar value

    Returns:
    ----
    A new Node instance created by Op.

    """
    return AddmmOp(node_A, node_B, node_C, alpha, beta, ctx=ctx)


def addmm_gradient_op(node_input, node_grad, beta=1.0, ctx=None):
    """Make a new instance of AddmmGradientOp and call the instance.

    Parameters:
    ----
    node_input : Node
        Input node.
    node_grad : Node
        Input node.
    beta : Scalar value

    Returns:
    ----
    A new Node instance created by Op.

    """
    return AddmmGradientOp(node_input, node_grad, beta, ctx=ctx)
