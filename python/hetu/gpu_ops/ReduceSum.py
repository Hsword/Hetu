from __future__ import absolute_import
import numpy as np
from .Node import Op
from ..gpu_links import reduce_sum


class ReduceSumOp(Op):
    def __init__(self, node_A, axes, keepdims=False, ctx=None):
        super().__init__(ReduceSumOp, [node_A], ctx)
        self.ori_status = None
        self.tar_status = None
        self.grad_node = None
        if axes is not None:
            if isinstance(axes, int):
                axes = [axes]
            self.axes = list(axes)
            assert all(map(lambda x: isinstance(x, int), self.axes))
        if keepdims is not None:
            if keepdims is True or keepdims is False:
                self.keepdims = [keepdims] * len(self.axes)
            else:
                keepdims = list(keepdims)
                assert len(keepdims) == len(self.axes)
                assert all(map(lambda x: isinstance(x, bool), keepdims))
                self.keepdims = keepdims

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.naive_copy:
            input_vals[0].copyto(output_val)
        else:
            if self.on_cpu:
                if all(self.keepdims) or not any(self.keepdims):
                    output_val[:] = np.sum(input_vals[0].asnumpy(), axis=tuple(
                        self.axes), keepdims=self.keepdims[0])
                else:
                    temp = input_vals[0].asnumpy()
                    for i in range(len(self.keepdims))[::-1]:
                        temp = np.sum(
                            temp, self.axes[i], keepdims=self.keepdims[i])
                    output_val[:] = temp
            else:
                reduce_sum(input_vals[0], output_val, self.axes, stream_handle)

    def gradient(self, output_grad):
        self.grad_set = False
        from .BroadcastShape import broadcast_shape_op
        self.grad_node = broadcast_shape_op(
            output_grad, None, None, ctx=self.raw_ctx)
        return [self.grad_node]

    def infer_shape(self, input_shapes):
        assert self.axes is not None and self.keepdims is not None
        assert len(input_shapes) == 1
        input_shape = list(input_shapes[0])
        for i in range(len(self.axes)):
            if self.axes[i] < 0:
                self.axes[i] += len(input_shape)
            assert 0 <= self.axes[i] < len(input_shape)
        if self.grad_node is not None:
            self.grad_node.target_shape = tuple(input_shape)
            add_axes = []
            for i in range(len(self.axes)):
                if not self.keepdims[i]:
                    add_axes.append(self.axes[i])
            self.grad_node.add_axes = add_axes
        for i in range(len(self.axes)):
            input_shape[self.axes[i]] = 1 if self.keepdims[i] else 0
        input_shape = [x for x in input_shape if x > 0]
        if input_shape == []:
            result = (1,)
        else:
            result = tuple(input_shape)
        from_size = np.prod(input_shapes[0], dtype=int)
        to_size = np.prod(result, dtype=int)
        self.naive_copy = (from_size == to_size)
        return result

    def forward_deduce_states(self, input_statuses, status, deduce_order):
        assert len(input_statuses) == len(self.inputs)
        if deduce_order:
            if self.ori_status is not None:
                status.copy_order_from(self.ori_status)
            elif input_statuses[0].order is not None:
                order = list(input_statuses[0].order)
                dup_occur = 0
                prev_dup = False
                partial_candidate = self.axes + [-2]
                for i, o in list(enumerate(order))[::-1]:
                    if o in partial_candidate:
                        if not prev_dup:
                            dup_occur += 1
                        prev_dup = True
                        if o != -2:
                            if -2 not in order:
                                order[i] = -2
                            else:
                                order.pop(i)
                    else:
                        prev_dup = False
                assert dup_occur <= 1, 'Duplicate dimension and reduce dimensions must be consecutive!'
                for i in range(len(order)):
                    order[i] -= sum([x < order[i]
                                     for j, x in enumerate(self.axes) if not self.keepdims[j]])
                status.set_order(tuple(order))
        else:
            if self.ori_status is not None:
                status.copy_state_from(self.ori_status)
            else:
                if input_statuses[0].valid_state():
                    state, duplicate = input_statuses[0].get()
                    partial = input_statuses[0].partial if input_statuses[0].enable_partial else 1
                    state = dict(state)
                    for k in self.axes:
                        if k in state:
                            partial *= state.pop(k)
                    for k in sorted(state.keys()):
                        new_k = k - \
                            sum([x < k for j, x in enumerate(
                                self.axes) if not self.keepdims[j]])
                        if new_k != k:
                            state[new_k] = state.pop(k)
                    status.set_state(state, duplicate, partial)

    def backward_deduce_states(self, status, input_statuses, deduce_order):
        assert len(input_statuses) == len(self.inputs)
        if self.grad_node is not None and not self.grad_set:
            self.grad_node.ori_status = input_statuses[0]
            self.grad_node.tar_status = status.remove_partial()
            self.grad_set = True
        if deduce_order:
            if self.tar_status is not None:
                input_statuses[0].copy_order_from(self.tar_status)
        else:
            if self.tar_status is not None:
                input_statuses[0].copy_state_from(self.tar_status)

    def reset_status(self):
        self.ori_status = None
        self.tar_status = None
        self.grad_set = False


def reduce_sum_op(node, axes, keepdims=False, ctx=None):
    """Creates a node that represents np.sum(node_A, axis, keepdims).

    Parameters:
    ----
    node : Node
        The Node needed to be summed.
    axes : int or list
        The axis/axes needed to be summed.
    keepdims: bool or list
        Whether to keep the dimension(s).

    Returns:
    ----
    A new Node instance created by Op.

    """
    return ReduceSumOp(node, axes, keepdims, ctx=ctx)
