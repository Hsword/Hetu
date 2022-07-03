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
        matmul_forward_deduce(input_statuses, status, deduce_order,
                              self.matmul_attr_trans_A, self.matmul_attr_trans_B)

    def backward_deduce_states(self, status, input_statuses, deduce_order):
        assert len(input_statuses) == 2
        matmul_backward_deduce(status, input_statuses, deduce_order,
                               self.matmul_attr_trans_A, self.matmul_attr_trans_B)

    def deduce_generated_backward_nodes_states(self, input_statuses, status, index):
        assert index is not None
        if index == -1:
            return status.remove_partial()
        else:
            return matmul_make_backward_status(status, self.matmul_attr_trans_A, self.matmul_attr_trans_B, index)


def matmul_forward_deduce(input_statuses, status, deduce_order, transA, transB):
    l2res_map = [
        {-1: 1, 0: 0, 1: -2},  # no trans
        {-1: 1, 1: 0, 0: -2},  # trans A
    ][transA]
    r2res_map = [
        {-1: 0, 0: -2, 1: 1},  # no trans
        {-1: 0, 0: 1, 1: -2},  # trans B
    ][transB]
    lstatus = input_statuses[0]
    rstatus = input_statuses[1]
    if deduce_order:
        if lstatus.valid_all() and rstatus.valid_all():
            lorder = lstatus.order
            rorder = rstatus.order
            new_lorder = list(l2res_map[x] for x in lorder)
            new_rorder = list(r2res_map[x] for x in rorder)
            if new_lorder != new_rorder:
                new_lorder[new_lorder.index(1)] = -1
                new_rorder[new_rorder.index(0)] = -1
                assert new_lorder == new_rorder
            elif 0 in new_lorder and lstatus.duplicate > 1:
                ind0 = new_lorder.index(0)
                ind1 = new_lorder.index(1)
                if ind0 > ind1:
                    ind0, ind1 = ind1, ind0
                assert ind0 + 1 == ind1
                new_lorder.insert(ind1, -1)
            status.set_order(tuple(new_lorder))
    else:
        if lstatus.valid_state() and rstatus.valid_state():
            lstate = lstatus.state
            rstate = rstatus.state
            lrow = lstate.get(int(transA), 1)
            lcol = lstate.get(1-transA, 1)
            rrow = rstate.get(int(transB), 1)
            rcol = rstate.get(1-transB, 1)
            assert lcol == rrow
            res_state = (lrow, rcol)
            res_partial = lcol
            status.set_state(res_state, partial=res_partial)


def matmul_backward_deduce(status, input_statuses, deduce_order, transA, transB):
    res2l_map = [
        {-2: 1, 0: 0, 1: -1, -1: -1},  # no trans
        {-2: 0, 0: 1, 1: -1, -1: -1},  # trans A
    ][transA]
    res2r_map = [
        {-2: 0, 0: -1, 1: 1, -1: -1},  # no trans
        {-2: 1, 0: -1, 1: 0, -1: -1},  # trans B
    ][transB]
    if deduce_order:
        if status.valid_all():
            res_order = tuple(res2l_map[x] for x in status.order)
            input_statuses[0].set_order(res_order)
            res_order = tuple(res2r_map[x] for x in status.order)
            input_statuses[1].set_order(res_order)
    else:
        if status.valid_state():
            state, duplicate = status.get()
            partial = status.partial
            state = dict(state)
            state[-1] = duplicate
            state[-2] = partial if status.enable_partial else 1
            lstate, lduplicate = {}, 1
            rstate, rduplicate = {}, 1
            for key, value in state.items():
                lkey = res2l_map[key]
                if lkey == -1:
                    lduplicate *= value
                else:
                    lstate[lkey] = value
                rkey = res2r_map[key]
                if rkey == -1:
                    rduplicate *= value
                else:
                    rstate[rkey] = value
            input_statuses[0].set_state(lstate, lduplicate)
            input_statuses[1].set_state(rstate, rduplicate)


def matmul_make_backward_status(status, transA, transB, index):
    from ..context import NodeStatus
    new_status = NodeStatus(dev_num=status.dev_num, partial_or_node=True)
    if index < 2:
        res2genl_map = [
            {-2: 1, 0: 0, 1: -2, -1: -1},  # no trans
            {-2: 0, 0: 1, 1: -2, -1: -1},  # trans A
        ][transA]
        res2genr_map = [
            {-2: 0, 0: -2, 1: 1, -1: -1},  # no trans
            {-2: 1, 0: -2, 1: 0, -1: -1},  # trans B
        ][transB]
        res_map = res2genl_map if index == 0 else res2genr_map
        all_state = dict(status.state)
        all_state[-2] = 1 if status.partial is None else status.partial
        all_state[-1] = status.duplicate
        res_state = {}
        for key in [-2, -1, 0, 1]:
            value = all_state.get(key, 1)
            res_state[res_map[key]] = value
        partial = res_state.pop(-2)
        duplicate = res_state.pop(-1)
        new_status.set_state(res_state, duplicate, partial)
        order = tuple(res_map[x] for x in status.order)
        new_status.set_order(order)
    else:
        # for bias in linearop
        assert index == 2
        new_status.set_state(*status.combine_state((-2, -1), (0, -2), (1, 0)))
        new_status.set_order(status.combine_order((-2, -1), (0, -2), (1, 0)))
    return new_status


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
