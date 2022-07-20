from __future__ import absolute_import
import numpy as np
from .Node import Op
from .ReduceSum import reduce_sum_op
from ..gpu_links import broadcast_shape_simple
from .. import ndarray


class BroadcastShapeOp(Op):
    def __init__(self, node_A, shape, add_axes=(), ctx=None):
        super().__init__(BroadcastShapeOp, [node_A], ctx)
        self.target_shape = shape
        self.add_axes = add_axes
        self.ori_status = None
        self.tar_status = None
        self.grad_node = None
        self.grad_set = False

    def compute(self, input_vals, output_val, stream_handle=None):
        assert self.target_shape is not None and self.add_axes is not None
        if self.on_cpu:
            input_shape = list(input_vals[0].shape)
            for i in range(len(input_shape)):
                if self.add_axes and i in self.add_axes:
                    input_shape[i] = 1
            output_val[:] = np.broadcast_to(
                input_vals[0].asnumpy().reshape(input_shape), self.target_shape)
        else:
            if self.inplace:
                input_vals[0].broadcast_to(
                    self.target_shape, output_val, self.add_axes)
            else:
                # broadcast_shape(input_vals[0], output_val, self.add_axes, stream_handle)
                broadcast_shape_simple(
                    input_vals[0], output_val, self.out_strides, self.in_dims, stream_handle)

    def gradient(self, output_grad):
        self.grad_node = reduce_sum_op(
            output_grad, None, None, ctx=self.raw_ctx)
        return [self.grad_node]

    def infer_shape(self, input_shapes):
        assert self.target_shape is not None and self.add_axes is not None
        assert len(input_shapes) == 1
        input_shape = list(input_shapes[0])
        output_shape = list(self.target_shape)
        output_ndim = len(output_shape)
        assert len(input_shape) <= output_ndim
        diff = output_ndim - len(input_shape)
        if self.add_axes:
            assert diff == len(self.add_axes) or input_shape == [1]
            assert all([axis < output_ndim for axis in self.add_axes])
            in_ind = 0
            for i in range(output_ndim):
                if i not in self.add_axes:
                    assert input_shape[in_ind] == output_shape[i]
                    in_ind += 1
            if self.grad_node is not None:
                self.grad_node.axes = tuple(self.add_axes)
                self.grad_node.keepdims = [False] * len(self.add_axes)
        else:
            axes = list(range(diff))
            keepdims = [False] * diff
            input_shape = [1] * diff + input_shape
            for i in range(output_ndim):
                if output_shape[i] == -1:
                    output_shape[i] = input_shape[i]
                assert output_shape[i] > 0 and isinstance(output_shape[i], int)
                assert input_shape[i] == 1 or input_shape[i] == output_shape[i]
                if i >= diff and input_shape[i] == 1 and output_shape[i] > 1:
                    axes.append(i)
                    keepdims.append(True)
            if self.grad_node is not None:
                self.grad_node.axes = axes
                self.grad_node.keepdims = keepdims

        # here we save the output strides and input dimensions for GPU computation
        if self.on_gpu and not self.inplace:
            input_shape = list(input_shapes[0])
            out_strides = [0 for _ in range(output_ndim)]
            temp_size = 1
            for i in range(output_ndim - 1, -1, -1):
                out_strides[i] = temp_size
                temp_size *= output_shape[i]
            if self.add_axes:
                in_dims = [0 for _ in range(output_ndim)]
                for i in range(diff):
                    in_dims[self.add_axes[i]] = 1
                temp_ind = 0
                for dim in input_shape:
                    while in_dims[temp_ind] == 1:
                        temp_ind += 1
                    in_dims[temp_ind] = dim
                    temp_ind += 1
            else:
                in_dims = [1 for _ in range(diff)] + input_shape

            self.out_strides = ndarray.array(
                out_strides, self.ctx, dtype=np.int32)
            self.in_dims = ndarray.array(in_dims, self.ctx, dtype=np.int32)
        return tuple(output_shape)

    def naive_infer_shape(self, input_shapes):
        assert self.target_shape is not None and self.add_axes is not None
        assert len(input_shapes) == 1
        input_shape = list(input_shapes[0])
        output_shape = list(self.target_shape)
        output_ndim = len(output_shape)
        assert len(input_shape) <= output_ndim
        diff = output_ndim - len(input_shape)
        if self.add_axes:
            assert diff == len(self.add_axes) or input_shape == [1]
            assert all([axis < output_ndim for axis in self.add_axes])
            in_ind = 0
            for i in range(output_ndim):
                if i not in self.add_axes:
                    assert input_shape[in_ind] == output_shape[i]
                    in_ind += 1
            if self.grad_node is not None:
                self.grad_node.axes = tuple(self.add_axes)
                self.grad_node.axes.keepdims = [False] * len(self.add_axes)
        else:
            axes = list(range(diff))
            keepdims = [False] * diff
            input_shape = [1] * diff + input_shape
            for i in range(output_ndim):
                if output_shape[i] == -1:
                    output_shape[i] = input_shape[i]
                assert output_shape[i] > 0 and isinstance(output_shape[i], int)
                assert input_shape[i] == 1 or input_shape[i] == output_shape[i]
                if i >= diff and input_shape[i] == 1 and output_shape[i] > 1:
                    axes.append(i)
                    keepdims.append(True)
            if self.grad_node is not None:
                self.grad_node.axes = axes
                self.grad_node.keepdims = keepdims

        return tuple(output_shape)

    def backward_hook(self, config):
        self.inplace = config.enable_lazy and self not in config.eval_node_list

    def forward_deduce_states(self, input_statuses, status, deduce_order):
        assert len(input_statuses) == len(self.inputs)
        if deduce_order:
            if self.ori_status is not None:
                status.copy_order_from(self.ori_status)
            else:
                # only support data parallel
                order = input_statuses[0].order
                if order is not None:
                    input_statuses[0].check_state(1, deduce_order)
                    status.set_order(order)
        else:
            if self.ori_status is not None:
                status.copy_state_from(self.ori_status)
            else:
                # only support data parallel
                state, duplicate = input_statuses[0].get()
                if state is not None:
                    input_statuses[0].check_state(1, deduce_order)
                status.set_state(state, duplicate)

    def backward_deduce_states(self, status, input_statuses, deduce_order):
        assert len(input_statuses) == len(self.inputs)
        if self.grad_node is not None and not self.grad_set:
            self.grad_node.ori_status = input_statuses[0]
            self.grad_node.tar_status = status
            self.grad_set = True
        if deduce_order:
            if self.tar_status is not None:
                input_statuses[0].copy_order_from(self.tar_status)
            else:
                # only support data parallel
                order = status.order
                if order is not None:
                    status.check_state(1, deduce_order)
                    input_statuses[0].set_order(order)
        else:
            if self.tar_status is not None:
                input_statuses[0].copy_state_from(self.tar_status)
            else:
                # only support data parallel
                state, duplicate = status.get()
                if state is not None:
                    status.check_state(1, deduce_order)
                input_statuses[0].set_state(state, duplicate)

    def reset_status(self):
        self.grad_set = False
        self.ori_status = None
        self.tar_status = None


def broadcast_shape_op(node_A, shape, add_axes=(), ctx=None):
    """Creates a node that represents np.broadcast_to(node_A, shape).

    Parameters:
    ----
    node_a : Node
        The Node to be broadcast.
    shape : tuple
        Target shape.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return BroadcastShapeOp(node_A, shape, add_axes=add_axes, ctx=ctx)
