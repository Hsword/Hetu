from __future__ import absolute_import
import numpy as np
from .Node import Op
from .ReduceSum import reduce_sum_op
from ..gpu_links import broadcast_shape_simple
from .. import ndarray


class BroadcastToOp(Op):
    def __init__(self, node_A, node_B, ctx=None):
        super().__init__(BroadcastToOp, [node_A, node_B], ctx)
        self.grad_node = None
        self.grad_set = False

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            input_shape = list(input_vals[1].shape)
            output_val[:] = np.broadcast_to(
                input_vals[0].asnumpy(), input_shape)
        else:
            if self.inplace:
                input_vals[0].broadcast_to(input_vals[1].shape, output_val)
            else:
                # broadcast_shape(input_vals[0], output_val, None, stream_handle)
                broadcast_shape_simple(
                    input_vals[0], output_val, self.out_strides, self.in_dims, stream_handle)

    def gradient(self, output_grad):
        self.grad_node = reduce_sum_op(
            output_grad, None, None, ctx=self.raw_ctx)
        return [self.grad_node, None]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 2
        input_shape = list(input_shapes[0])
        output_shape = list(input_shapes[1])
        output_ndim = len(output_shape)
        assert len(input_shape) <= output_ndim
        diff = output_ndim - len(input_shape)
        axes = list(range(diff))
        keepdims = [False] * diff
        input_shape = [1] * diff + input_shape
        for i in range(output_ndim):
            if isinstance(output_shape[i], (np.int32, np.int64)):
                output_shape[i] = output_shape[i].item()
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
            in_dims = [1 for _ in range(diff)] + input_shape

            self.out_strides = ndarray.array(
                out_strides, self.ctx, dtype=np.int32)
            self.in_dims = ndarray.array(in_dims, self.ctx, dtype=np.int32)
        return input_shapes[1]

    def naive_infer_shape(self, input_shapes):
        assert len(input_shapes) == 2
        input_shape = list(input_shapes[0])
        output_shape = list(input_shapes[1])
        output_ndim = len(output_shape)
        assert len(input_shape) <= output_ndim
        diff = output_ndim - len(input_shape)
        axes = list(range(diff))
        keepdims = [False] * diff
        input_shape = [1] * diff + input_shape
        for i in range(output_ndim):
            assert output_shape[i] > 0 and isinstance(output_shape[i], int)
            assert input_shape[i] == 1 or input_shape[i] == output_shape[i]
            if i >= diff and input_shape[i] == 1 and output_shape[i] > 1:
                axes.append(i)
                keepdims.append(True)
        if self.grad_node is not None:
            self.grad_node.axes = axes
            self.grad_node.keepdims = keepdims

        return input_shapes[1]

    def backward_hook(self, config):
        self.inplace = config.enable_lazy and self not in config.eval_node_list

    def forward_deduce_states(self, input_statuses, status, deduce_order):
        assert len(input_statuses) == len(self.inputs)
        status.copy_from(input_statuses[1], deduce_order)

    def backward_deduce_states(self, status, input_statuses, deduce_order):
        assert len(input_statuses) == len(self.inputs)
        if self.grad_node is not None and not self.grad_set:
            self.grad_node.ori_status = input_statuses[0]
            self.grad_node.tar_status = status
            self.grad_set = True
        # there is no information for input[0] here, so we don't deduce
        input_statuses[1].copy_from(status, deduce_order)

    def reset_status(self):
        self.grad_set = False


def broadcastto_op(node_A, node_B, ctx=None):
    """Creates a node that represents np.broadcast_to(node_A, node_B.shape).

    Parameters:
    ----
    node_a : Node
        The Node to be broadcast.
    node_b : Node
        Another Node with the target shape.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return BroadcastToOp(node_A, node_B, ctx=ctx)
