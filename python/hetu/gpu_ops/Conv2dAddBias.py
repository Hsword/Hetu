from __future__ import absolute_import
import numpy as np

from .Node import Op
from .Conv2d import conv2d_gradient_of_data_op, conv2d_gradient_of_filter_op
from .ReduceSum import reduce_sum_op
from ..gpu_links import CuDNN_conv2d_with_bias


class Conv2dAddBiasOp(Op):
    def __init__(self, node_A, node_B, bias, padding=0, stride=1, ctx=None):
        super().__init__(Conv2dAddBiasOp, [node_A, node_B, bias], ctx)
        if not isinstance(padding, tuple):
            assert isinstance(padding, int)
            padding = (padding, padding)
        self.padding = padding
        if not isinstance(stride, tuple):
            assert isinstance(stride, int)
            stride = (stride, stride)
        self.stride = stride

    def im2col(self, X, filter_H, filter_W, padding, stride):
        N, C, H, W = X.shape
        assert (H + 2 * padding[0] - filter_H) % stride[0] == 0
        assert (W + 2 * padding[1] - filter_W) % stride[1] == 0
        out_H = (H + 2 * padding[0] - filter_H) // stride[0] + 1
        out_W = (W + 2 * padding[1] - filter_W) // stride[1] + 1

        y_row_size = C * filter_H * filter_W
        y_col_size = out_H * out_W
        y_shape = (N, y_row_size, y_col_size)
        Y = np.empty(y_shape, dtype=X.dtype)

        for batch_index in range(N):
            for col_index in range(y_col_size):
                out_y = col_index // out_W
                out_x = col_index % out_W
                in_y = out_y * stride[0] - padding[0]
                in_x = out_x * stride[1] - padding[1]
                row_idx = 0
                for c in range(0, C):
                    for y in range(in_y, in_y + filter_H):
                        for x in range(in_x, in_x + filter_W):
                            if (x < 0 or x >= W or y < 0 or y >= H):
                                Y[batch_index, row_idx, col_index] = 0
                            else:
                                Y[batch_index, row_idx,
                                    col_index] = X[batch_index, c, y, x]
                            row_idx += 1
        return Y

    def np_conv2d(self, X, Filter, padding=(0, 0), stride=(1, 1)):
        """Implement a conv2d as a matrix multiply after im2col."""
        filter_outChannel, filter_inChannel, filter_H, filter_W = Filter.shape
        N, C, H, W = X.shape
        assert (H + 2 * padding[0] - filter_H) % stride[0] == 0
        assert (W + 2 * padding[1] - filter_W) % stride[1] == 0
        out_H = (H + 2 * padding[0] - filter_H) // stride[0] + 1
        out_W = (W + 2 * padding[1] - filter_W) // stride[1] + 1

        im2col_matrix = self.im2col(X, filter_H, filter_W, padding, stride)
        filter_matrix = Filter.reshape(filter_outChannel, -1)
        return np.matmul(filter_matrix, im2col_matrix).reshape(N, filter_outChannel, out_H, out_W)

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            output_val[:] = self.np_conv2d(
                input_vals[0].asnumpy(), input_vals[1].asnumpy(), self.padding, self.stride) +\
                input_vals[2].asnumpy().reshape((input_vals[2].shape[0], 1, 1))
        else:
            CuDNN_conv2d_with_bias(input_vals[0], input_vals[1], input_vals[2],
                                   output_val, self.padding, self.stride, stream_handle)

    def gradient(self, output_grad):
        return [conv2d_gradient_of_data_op(self.inputs[1], output_grad, self.inputs[0], self.padding, self.stride, ctx=self.raw_ctx),
                conv2d_gradient_of_filter_op(
                    self.inputs[0], output_grad, self.inputs[1], self.padding, self.stride, ctx=self.raw_ctx),
                reduce_sum_op(output_grad, [0, 2, 3], ctx=self.raw_ctx)]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 3
        N, _, H, W = input_shapes[0]
        f_O, _, f_H, f_W = input_shapes[1]
        assert len(input_shapes[2]) == 1 and input_shapes[2][0] == f_O
        padding = self.padding
        stride = self.stride
        filter_H = input_shapes[1][2]
        filter_W = input_shapes[1][3]
        out_H = (H + 2 * padding[0] - filter_H) // stride[0] + 1
        out_W = (W + 2 * padding[1] - filter_W) // stride[1] + 1
        return (N, f_O, out_H, out_W)

    def forward_deduce_states(self, input_statuses, status, deduce_order):
        assert len(input_statuses) == len(self.inputs)
        l2res_map = {0: 0, 1: -2, -1: 1}
        r2res_map = {-1: 0, 0: 1, 1: -2}
        conv2d_forward_deduce_states(
            input_statuses, status, deduce_order, l2res_map, r2res_map)

    def backward_deduce_states(self, status, input_statuses, deduce_order):
        assert len(input_statuses) == len(self.inputs)
        res2l_map = {0: 0, 1: -1, -2: 1, -1: -1}
        res2r_map = {-2: 1, 0: -1, 1: 0, -1: -1}
        conv2d_backward_deduce_states(
            status, input_statuses, deduce_order, res2l_map, res2r_map)
        if deduce_order:
            if status.valid_all():
                input_statuses[2].set_order(
                    status.combine_order(([0, -2], -1), (1, 0)))
        else:
            if status.valid_state():
                input_statuses[2].set_state(
                    *status.combine_state(([0, -2], -1), (1, 0)))

    def deduce_generated_backward_nodes_states(self, input_statuses, status, index):
        assert index is not None
        if index == -1:
            return status.remove_partial()
        else:
            from .Conv2d import conv2d_make_backward_status
            return conv2d_make_backward_status(status, index)


def conv2d_add_bias_op(node_A, node_B, bias, padding=0, stride=1, ctx=None):
    """Conv2d-with-bias node.

    Parameters:
    ----
    node_A : Node
        Input data node.
    node_B : Node
        Input filter node.
    bias : Node
        Bias node.
    padding :
        Padding size.
    stride :
        Stride size.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return Conv2dAddBiasOp(node_A, node_B, bias, padding, stride, ctx=ctx)
