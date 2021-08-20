from __future__ import absolute_import
import numpy as np
from .Node import Op
from ..gpu_links import reduce_sum


class ReduceSumOp(Op):
    def __init__(self, node_A, axes, keepdims=False, ctx=None):
        super().__init__(ReduceSumOp, [node_A], ctx)
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
        if hasattr(self, 'grad_node'):
            self.grad_node.target_shape = tuple(input_shape)
            add_axes = []
            for i in range(len(self.axes)):
                if not self.keepdims[i]:
                    add_axes.append(self.axes[i])
            self.grad_node.add_axes = add_axes
        for i in range(len(self.axes)):
            if self.axes[i] < 0:
                self.axes[i] += len(input_shape)
            assert 0 <= self.axes[i] < len(input_shape)
            input_shape[self.axes[i]] = 1 if self.keepdims[i] else 0
        input_shape = [x for x in input_shape if x > 0]
        if input_shape == []:
            return (1,)
        else:
            return tuple(input_shape)

    def forward_deduce_states(self, input_statuses, status, deduce_order):
        assert len(input_statuses) == len(self.inputs)
        if deduce_order:
            if hasattr(self, 'ori_status'):
                status.copy_order_from(self.ori_status)
            elif input_statuses[0].order is not None:
                order = list(input_statuses[0].order)
                for ax, kd in sorted(zip(self.axes, self.keepdims))[::-1]:
                    assert order[ax+1] == ax
                    if not kd:
                        order.pop(ax+1)
                cur = -1
                for ind in range(-1, len(order) - 1):
                    while cur not in order:
                        cur += 1
                    order[order.index(cur)] = ind
                    cur += 1
                status.set_order(tuple(order))
        else:
            if hasattr(self, 'ori_status'):
                status.copy_state_from(self.ori_status)
            else:
                state, duplicate = input_statuses[0].get()
                if state is not None:
                    state = list(state)
                    for ax, kd in sorted(zip(self.axes, self.keepdims))[::-1]:
                        assert state[ax] == 1
                        if not kd:
                            state.pop(ax)
                    status.set_state(tuple(state), duplicate)

    def backward_deduce_states(self, status, input_statuses, deduce_order):
        assert len(input_statuses) == len(self.inputs)
        if hasattr(self, 'grad_node') and not self.grad_set:
            self.grad_node.ori_status = input_statuses[0]
            self.grad_node.tar_status = status
            self.grad_set = True
        if deduce_order:
            if hasattr(self, 'tar_status'):
                input_statuses[0].copy_order_from(self.tar_status)
            elif status.order is not None:
                order = list(status.order)
                min_ax = min(self.axes)
                for kd in self.keepdims:
                    if not kd:
                        order += [len(order) - 1]
                assert order[min_ax + 1:] == list(range(min_ax, len(order)-1))
                input_statuses[0].set_order(tuple(order))
        else:
            if hasattr(self, 'tar_status'):
                input_statuses[0].copy_state_from(self.tar_status)
            else:
                state, duplicate = status.get()
                if state is not None:
                    state = list(state)
                    for ax, kd in sorted(zip(self.axes, self.keepdims)):
                        if kd:
                            assert state[ax] == 1
                        else:
                            state.insert(ax, 1)
                    input_statuses[0].set_state(tuple(state), duplicate)


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
