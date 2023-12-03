from __future__ import absolute_import
import numpy as np
from .Node import Op
from .._base import DNNL_LIB
from ..cpu_links import transpose as cpu_transpose
from ..gpu_links import matrix_transpose_simple
from .. import ndarray


class TransposeOp(Op):
    def __init__(self, node_A, perm=None, ctx=None):
        super().__init__(TransposeOp, [node_A], ctx)
        self.perm = perm

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            if DNNL_LIB['cpu_Transpose']:
                cpu_transpose(input_vals[0], output_val, self.perm)
            else:
                output_val[:] = np.transpose(
                    input_vals[0].asnumpy(), self.perm)
        else:
            # matrix_transpose(input_vals[0], output_val, self.perm, stream_handle)
            matrix_transpose_simple(
                input_vals[0], output_val, self.gpu_buffer, stream_handle)

    def gradient(self, output_grad):
        if self.perm:
            grad_perm = [0 for _ in self.perm]
            for i in range(len(self.perm)):
                grad_perm[self.perm[i]] = i
        else:
            grad_perm = None
        return [transpose_op(output_grad, grad_perm, ctx=self.raw_ctx)]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        # only support matrix transpose
        # assert len(input_shapes[0]) == 2
        ori_shape = list(input_shapes[0])
        if self.perm is None:
            self.perm = list(range(len(ori_shape))[::-1])
            res_shape = ori_shape[::-1]
        else:
            assert len(self.perm) == len(ori_shape) and set(
                self.perm) == set(range(len(self.perm)))
            res_shape = [ori_shape[self.perm[i]]
                         for i in range(len(ori_shape))]

        # here we save the information for GPU computation
        if self.on_gpu:
            ndim = len(ori_shape)
            buffer = [0 for _ in range(3 * ndim)]
            in_stride = 1
            out_stride = 1
            for i in range(ndim - 1, -1, -1):
                buffer[i] = in_stride
                buffer[ndim + i] = out_stride
                buffer[2 * ndim + i] = self.perm[i]
                in_stride *= ori_shape[i]
                out_stride *= res_shape[i]
            self.gpu_buffer = ndarray.array(
                buffer, self.ctx, dtype=np.uintc)
        return tuple(res_shape)

    def naive_infer_shape(self, input_shapes):
        assert len(input_shapes) == 1
        # only support matrix transpose
        # assert len(input_shapes[0]) == 2
        ori_shape = list(input_shapes[0])
        if self.perm is None:
            self.perm = list(range(len(ori_shape))[::-1])
            res_shape = ori_shape[::-1]
        else:
            assert len(self.perm) == len(ori_shape) and set(
                self.perm) == set(range(len(self.perm)))
            res_shape = [ori_shape[self.perm[i]]
                         for i in range(len(ori_shape))]
        return tuple(res_shape)

    def forward_deduce_states(self, input_statuses, status, deduce_order):
        assert len(input_statuses) == 1
        assert self.perm is not None, 'Deduction requires permutation to be specified.'
        if deduce_order:
            if input_statuses[0].valid_all():
                order = input_statuses[0].order
                if order is not None:
                    new_order = []
                    for o in order:
                        if o == -1:
                            new_order.append(o)
                        else:
                            new_order.append(self.perm.index(o))
                    status.set_order(tuple(new_order))
        else:
            if input_statuses[0].valid_state():
                state, duplicate = input_statuses[0].get()
                res_state = {self.perm.index(k): v for k, v in state.items()}
                status.set_state(res_state, duplicate)

    def backward_deduce_states(self, status, input_statuses, deduce_order):
        assert len(input_statuses) == 1
        assert self.perm is not None, 'Deduction requires permutation to be specified.'
        if deduce_order:
            if status.valid_all():
                order = status.order
                if order is not None:
                    new_order = []
                    for o in order:
                        if o == -1:
                            new_order.append(o)
                        else:
                            new_order.append(self.perm[o])
                    input_statuses[0].set_order(tuple(new_order))
        else:
            if status.valid_state():
                state, duplicate = status.get()
                res_state = {self.perm[k]: v for k, v in state.items()}
                input_statuses[0].set_state(res_state, duplicate)


def transpose_op(node_A, perm=None, ctx=None):
    """Make a new instance of transpose and call the instance.

    Parameters:
    ----
    node_A : Node
        Node to be transposed.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return TransposeOp(node_A, perm, ctx=ctx)
