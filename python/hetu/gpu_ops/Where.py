from __future__ import absolute_import
import numpy as np
from .Node import Op
from ..gpu_links import where, where_const


class WhereOp(Op):
    def __init__(self, cond, node_A, node_B, ctx=None):
        super().__init__(WhereOp, [cond, node_A, node_B], ctx)

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            output_val[:] = np.where(input_vals[0].asnumpy(
            ), input_vals[1].asnumpy(), input_vals[2].asnumpy())
        else:

            where(input_vals[0], input_vals[1],
                  input_vals[2], output_val, stream_handle)

    def gradient(self, output_grad):
        from .ZerosLike import zeroslike_op
        zeros = zeroslike_op(self.inputs[0], ctx=self.raw_ctx)
        grad_A = where_op(self.inputs[0], output_grad, zeros, ctx=self.raw_ctx)
        grad_B = where_op(self.inputs[0], zeros, output_grad, ctx=self.raw_ctx)
        return [None, grad_A, grad_B]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 3
        assert tuple(input_shapes[0]) == tuple(
            input_shapes[1]) == tuple(input_shapes[2])
        return input_shapes[0]


class WhereConstOp(Op):
    def __init__(self, cond, node_A, const_attr, ctx=None):
        super().__init__(WhereOp, [cond, node_A], ctx)
        self.const_attr = const_attr

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            output_val[:] = np.where(input_vals[0].asnumpy(
            ), input_vals[1].asnumpy(), self.const_attr)
        else:
            where_const(input_vals[0], input_vals[1],
                        self.const_attr, output_val, stream_handle)

    def gradient(self, output_grad):
        grad = where_const_op(
            self.inputs[0], output_grad, 0., ctx=self.raw_ctx)
        return [None, grad]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 2
        assert tuple(input_shapes[0]) == tuple(input_shapes[1])
        return input_shapes[0]


def where_op(cond, node_A, node_B, ctx=None):
    """Creates a node that represents np.where.

    Parameters:
    ----
    cond : Node of a condition array
    node_A : Node, output if cond
    node_B : Node, output if not cond

    Returns:
    ----
    A new Node instance created by Op.

    """
    return WhereOp(cond, node_A, node_B, ctx=ctx)


def where_const_op(cond, node_A, const_attr, ctx=None):
    """Creates a node that represents np.where.

    Parameters:
    ----
    cond : Node of a condition array
    node_A : Node, output if cond
    const_attr : float, output if not cond

    Returns:
    ----
    A new Node instance created by Op.

    """
    return WhereConstOp(cond, node_A, const_attr, ctx=ctx)
