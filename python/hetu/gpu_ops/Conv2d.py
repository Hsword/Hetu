from __future__ import absolute_import
import numpy as np
from .Node import Op
from .._base import DNNL_LIB
from ..cpu_links import conv2d as cpu_conv2d
from ..cpu_links import conv2d_gradient_of_data as cpu_conv2d_gradient_of_data
from ..cpu_links import conv2d_gradient_of_filter as cpu_conv2d_gradient_of_filter
from ..gpu_links import CuDNN_conv2d
from ..gpu_links import CuDNN_conv2d_gradient_of_data
from ..gpu_links import CuDNN_conv2d_gradient_of_filter


class Conv2dOp(Op):
    # nodeA : x  nodeB : filter
    def __init__(self, node_A, node_B, padding=0, stride=1, ctx=None):
        super().__init__(Conv2dOp, [node_A, node_B], ctx)
        self.padding = padding
        self.stride = stride

    def im2col(self, X, filter_H, filter_W, padding, stride):
        N, C, H, W = X.shape
        assert (H + 2 * padding - filter_H) % stride == 0
        assert (W + 2 * padding - filter_W) % stride == 0
        out_H = (H + 2 * padding - filter_H) // stride + 1
        out_W = (W + 2 * padding - filter_W) // stride + 1

        y_row_size = C * filter_H * filter_W
        y_col_size = out_H * out_W
        y_shape = (N, y_row_size, y_col_size)
        Y = np.empty(y_shape, dtype=X.dtype)

        for batch_index in range(N):
            for col_index in range(y_col_size):
                out_y = col_index // out_W
                out_x = col_index % out_W
                in_y = out_y * stride - padding
                in_x = out_x * stride - padding
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

    def np_conv2d(self, X, Filter, padding=0, stride=1):
        """Implement a conv2d as a matrix multiply after im2col."""
        filter_outChannel, filter_inChannel, filter_H, filter_W = Filter.shape
        N, C, H, W = X.shape
        assert (H + 2 * padding - filter_H) % stride == 0
        assert (W + 2 * padding - filter_W) % stride == 0
        out_H = (H + 2 * padding - filter_H) // stride + 1
        out_W = (W + 2 * padding - filter_W) // stride + 1

        im2col_matrix = self.im2col(X, filter_H, filter_W, padding, stride)
        filter_matrix = Filter.reshape(filter_outChannel, -1)
        return np.matmul(filter_matrix, im2col_matrix).reshape(N, filter_outChannel, out_H, out_W)

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            if DNNL_LIB['DnnlConv2d']:
                cpu_conv2d(input_vals[0], input_vals[1],
                           output_val, self.padding, self.stride)
            else:
                output_val[:] = self.np_conv2d(
                    input_vals[0].asnumpy(), input_vals[1].asnumpy(), self.padding, self.stride)
        else:
            CuDNN_conv2d(input_vals[0], input_vals[1],
                         output_val, self.padding, self.stride, stream_handle)

    def gradient(self, output_grad):
        return [conv2d_gradient_of_data_op(self.inputs[1], output_grad, self.padding, self.stride, ctx=self.raw_ctx),
                conv2d_gradient_of_filter_op(self.inputs[0], output_grad, self.padding, self.stride, ctx=self.raw_ctx)]

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 2
        N, _, H, W = input_shapes[0]
        f_O, _, f_H, f_W = input_shapes[1]
        padding = self.padding
        stride = self.stride
        filter_H = input_shapes[1][2]
        filter_W = input_shapes[1][3]
        out_H = (H + 2 * padding - filter_H) // stride + 1
        out_W = (W + 2 * padding - filter_W) // stride + 1
        return (N, f_O, out_H, out_W)

    def forward_deduce_states(self, input_statuses, status, deduce_order):
        assert len(input_statuses) == len(self.inputs)
        l2res_map = {0: 0, 1: -1, -1: 1, 2: 2, 3: 3}
        r2res_map = {-1: 0, 0: 1, 1: -1, 2: 2, 3: 3}
        res2l_map = {0: 0, -1: 1, 1: -1, 2: 2, 3: 3}
        res2r_map = {0: -1, -1: 1, 1: 0, 2: 2, 3: 3}
        if deduce_order:
            if input_statuses[0].valid_all():
                order = input_statuses[0].order
                status.set_order(tuple(l2res_map[x] for x in order))
            elif input_statuses[1].valid_all():
                order = input_statuses[1].order
                status.set_order(tuple(r2res_map[x] for x in order))
        else:
            if input_statuses[0].valid_state():
                state, duplicate = input_statuses[0].get()
                keys = dict(state)
                keys[-1] = duplicate
                res_state = tuple(keys.get(res2l_map[x], 1) for x in range(4))
                res_duplicate = keys.get(res2l_map[-1], 1)
                status.set_state(res_state, res_duplicate)
            elif input_statuses[1].valid_state():
                state, duplicate = input_statuses[1].get()
                keys = dict(state)
                keys[-1] = duplicate
                res_state = tuple(keys.get(res2r_map[x], 1) for x in range(4))
                res_duplicate = keys.get(res2r_map[-1], 1)
                status.set_state(res_state, res_duplicate)

    def backward_deduce_states(self, status, input_statuses, deduce_order):
        assert len(input_statuses) == len(self.inputs)
        l2res_map = {0: 0, 1: -1, -1: 1, 2: 2, 3: 3}
        r2res_map = {-1: 0, 0: 1, 1: -1, 2: 2, 3: 3}
        res2l_map = {0: 0, -1: 1, 1: -1, 2: 2, 3: 3}
        res2r_map = {0: -1, -1: 1, 1: 0, 2: 2, 3: 3}
        if deduce_order:
            if status.valid_all():
                res_order = tuple(res2l_map[x] for x in status.order)
                input_statuses[0].set_order(res_order)
                res_order = tuple(res2r_map[x] for x in status.order)
                input_statuses[1].set_order(res_order)
        else:
            if status.valid_state():
                state, duplicate = status.get()
                keys = dict(state)
                keys[-1] = duplicate
                res_state = tuple(keys.get(l2res_map[x], 1) for x in range(4))
                res_duplicate = keys.get(l2res_map[-1], 1)
                input_statuses[0].set_state(res_state, res_duplicate)
                res_state = tuple(keys.get(r2res_map[x], 1) for x in range(4))
                res_duplicate = keys.get(r2res_map[-1], 1)
                input_statuses[1].set_state(res_state, res_duplicate)
            else:
                if input_statuses[0].state is not None:
                    input_statuses[1].set_state(
                        None, input_statuses[0].state.get(0, 1))
                if input_statuses[1].state is not None:
                    input_statuses[0].set_state(
                        None, input_statuses[1].state.get(0, 1))


class Conv2d_Gradient_of_DataOp(Op):
    # nodeA : filter  nodeB : Y_gradient
    def __init__(self, node_A, node_B, padding=0, stride=1, ctx=None):
        super().__init__(Conv2d_Gradient_of_DataOp, [node_A, node_B], ctx)
        self.padding = padding
        self.stride = stride

    def im2col_transpose(self, N, C, H, W, filter_H, filter_W, Y, padding, stride):
        assert (H + 2 * padding - filter_H) % stride == 0
        assert (W + 2 * padding - filter_W) % stride == 0
        out_H = (H + 2 * padding - filter_H) // stride + 1
        out_W = (W + 2 * padding - filter_W) // stride + 1
        _, y_row_size, y_col_size = Y.shape

        der_X_shape = (N, C, H, W)
        der_X = np.zeros(der_X_shape, dtype=Y.dtype)

        for batch_index in range(N):
            for col_index in range(y_col_size):
                out_y = col_index // out_W
                out_x = col_index % out_W
                in_y = out_y * stride - padding
                in_x = out_x * stride - padding
                row_idx = 0
                for c in range(0, C):
                    for y in range(in_y, in_y + filter_H):
                        for x in range(in_x, in_x + filter_W):
                            if (x < 0 or x >= W or y < 0 or y >= H):
                                Y[batch_index, row_idx, col_index] = 0
                            else:
                                der_X[batch_index, c, y,
                                      x] += Y[batch_index, row_idx, col_index]
                            row_idx += 1
        return der_X

    def np_Conv2dGradient_data(self, X_N, X_C, X_H, X_W, Filter, Y, padding=0, stride=1):
        filter_outChannel, filter_inChannel, filter_H, filter_W = Filter.shape
        Y_N, Y_C, Y_H, Y_W = Y.shape
        YY = Y.reshape((Y_N, Y_C, Y_H * Y_W))    # transformed to im2col Y
        F_filter = Filter.reshape((filter_outChannel, -1))

        gradient_im2col_XX = np.matmul(F_filter.T, YY)
        gradient_X = self.im2col_transpose(
            X_N, X_C, X_H, X_W, filter_H, filter_W, gradient_im2col_XX, padding, stride)    # gradient of x
        return gradient_X

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            if DNNL_LIB['DnnlConv2d_Gradient_of_Data']:
                cpu_conv2d_gradient_of_data(
                    input_vals[0], input_vals[1], output_val, self.padding, self.stride)
            else:
                N = input_vals[1].shape[0]
                C = input_vals[0].shape[1]
                H = (input_vals[1].shape[2] - 1) * self.stride + \
                    input_vals[0].shape[2] - 2 * self.padding
                W = (input_vals[1].shape[3] - 1) * self.stride + \
                    input_vals[0].shape[3] - 2 * self.padding
                output_val[:] = self.np_Conv2dGradient_data(
                    N, C, H, W, input_vals[0].asnumpy(), input_vals[1].asnumpy(), padding=self.padding, stride=self.stride)
        else:
            CuDNN_conv2d_gradient_of_data(
                input_vals[0], input_vals[1], output_val, padding=self.padding, stride=self.stride, stream=stream_handle)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 2
        N = input_shapes[1][0]
        C = input_shapes[0][1]
        H = (input_shapes[1][2] - 1) * self.stride + \
            input_shapes[0][2] - 2 * self.padding
        W = (input_shapes[1][3] - 1) * self.stride + \
            input_shapes[0][3] - 2 * self.padding
        return (N, C, H, W)

    def forward_deduce_states(self, input_statuses, status, deduce_order):
        assert len(input_statuses) == len(self.inputs)
        l2res_map = {-1: 0, 0: -1, 1: 1, 2: 2, 3: 3}
        r2res_map = {-1: 1, 0: 0, 1: -1, 2: 2, 3: 3}
        res2l_map = {-1: 0, 0: -1, 1: 1, 2: 2, 3: 3}
        res2r_map = {-1: 1, 0: 0, 1: -1, 2: 2, 3: 3}
        if deduce_order:
            if input_statuses[0].valid_all():
                order = input_statuses[0].order
                status.set_order(tuple(l2res_map[x] for x in order))
            elif input_statuses[1].valid_all():
                order = input_statuses[1].order
                status.set_order(tuple(r2res_map[x] for x in order))
        else:
            if input_statuses[0].valid_state():
                state, duplicate = input_statuses[0].get()
                keys = dict(state)
                keys[-1] = duplicate
                res_state = tuple(keys.get(res2l_map[x], 1) for x in range(4))
                res_duplicate = keys.get(res2l_map[-1], 1)
                status.set_state(res_state, res_duplicate)
            elif input_statuses[1].valid_state():
                state, duplicate = input_statuses[1].get()
                keys = dict(state)
                keys[-1] = duplicate
                res_state = tuple(keys.get(res2r_map[x], 1) for x in range(4))
                res_duplicate = keys.get(res2r_map[-1], 1)
                status.set_state(res_state, res_duplicate)

    def backward_deduce_states(self, status, input_statuses, deduce_order):
        assert len(input_statuses) == len(self.inputs)
        l2res_map = {-1: 0, 0: -1, 1: 1, 2: 2, 3: 3}
        r2res_map = {-1: 1, 0: 0, 1: -1, 2: 2, 3: 3}
        res2l_map = {-1: 0, 0: -1, 1: 1, 2: 2, 3: 3}
        res2r_map = {-1: 1, 0: 0, 1: -1, 2: 2, 3: 3}
        if deduce_order:
            if status.valid_all():
                res_order = tuple(res2l_map[x] for x in status.order)
                input_statuses[0].set_order(res_order)
                res_order = tuple(res2r_map[x] for x in status.order)
                input_statuses[1].set_order(res_order)
        else:
            if status.valid_state():
                state, duplicate = status.get()
                keys = dict(state)
                keys[-1] = duplicate
                res_state = tuple(keys.get(l2res_map[x], 1) for x in range(4))
                res_duplicate = keys.get(l2res_map[-1], 1)
                input_statuses[0].set_state(res_state, res_duplicate)
                res_state = tuple(keys.get(r2res_map[x], 1) for x in range(4))
                res_duplicate = keys.get(r2res_map[-1], 1)
                input_statuses[1].set_state(res_state, res_duplicate)
            else:
                if input_statuses[0].state is not None:
                    input_statuses[1].set_state(
                        None, input_statuses[0].state.get(1, 1))
                if input_statuses[1].state is not None:
                    input_statuses[0].set_state(
                        None, input_statuses[1].state.get(0, 1))


class Conv2d_Gradient_of_FilterOp(Op):
    # nodeA : input_x  nodeB : gradient_Y
    def __init__(self, input_X, gradient_Y, padding=0, stride=1, ctx=None):
        super().__init__(Conv2d_Gradient_of_FilterOp,
                         [input_X, gradient_Y], ctx)
        self.padding = padding
        self.stride = stride

    def im2col(self, X, filter_H, filter_W, padding, stride):
        N, C, H, W = X.shape
        assert (H + 2 * padding - filter_H) % stride == 0
        assert (W + 2 * padding - filter_W) % stride == 0
        out_H = (H + 2 * padding - filter_H) // stride + 1
        out_W = (W + 2 * padding - filter_W) // stride + 1

        y_row_size = C * filter_H * filter_W
        y_col_size = out_H * out_W
        y_shape = (N, y_row_size, y_col_size)
        Y = np.empty(y_shape, dtype=X.dtype)

        for batch_index in range(N):
            for col_index in range(y_col_size):
                out_y = col_index // out_W
                out_x = col_index % out_W
                in_y = out_y * stride - padding
                in_x = out_x * stride - padding
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

    def np_Conv2dGradient_Filter(self, filter_outChannel, filter_inChannel, filter_H, filter_W, X, Y, padding=0, stride=1):
        """Implement a conv2d_transpose as a matrix multiply after im2col."""
        X_N, X_C, X_H, X_W = X.shape
        Y_N, Y_C, Y_H, Y_W = Y.shape
        YY = Y.reshape((Y_N, Y_C, Y_H * Y_W))    # transformed to im2col Y
        # XX = X.reshape((X_N, X_C, X_W * X_H))   # transformed to im2col X
        im2col_XX = self.im2col(X, filter_H, filter_W, padding, stride)
        gradient_filter = np.zeros(shape=(
            filter_outChannel, filter_inChannel * filter_H * filter_W), dtype=Y.dtype)

        for i in range(X_N):
            gradient_filter += np.matmul(YY[i], im2col_XX[i].T)
        gradient_filter = gradient_filter.reshape(
            (filter_outChannel, filter_inChannel, filter_H, filter_W))

        return gradient_filter

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            if DNNL_LIB['DnnlConv2d_Gradient_of_Filter']:
                cpu_conv2d_gradient_of_filter(
                    input_vals[0], input_vals[1], output_val, self.padding, self.stride)
            else:
                f_N = input_vals[1].shape[1]
                f_C = input_vals[0].shape[1]
                f_H = input_vals[1].shape[2] + 2 * self.padding - \
                    (input_vals[1].shape[2] - 1) * self.stride
                f_W = input_vals[1].shape[3] + 2 * self.padding - \
                    (input_vals[1].shape[3] - 1) * self.stride
                output_val[:] = self.np_Conv2dGradient_Filter(
                    f_N, f_C, f_H, f_W, input_vals[0].asnumpy(), input_vals[1].asnumpy(), padding=self.padding, stride=self.stride)
        else:
            CuDNN_conv2d_gradient_of_filter(
                input_vals[0], input_vals[1], output_val, padding=self.padding, stride=self.stride, stream=stream_handle)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 2
        f_N = input_shapes[1][1]
        f_C = input_shapes[0][1]
        f_H = input_shapes[0][2] + 2 * self.padding - \
            (input_shapes[1][2] - 1) * self.stride
        f_W = input_shapes[0][3] + 2 * self.padding - \
            (input_shapes[1][3] - 1) * self.stride

        return (f_N, f_C, f_H, f_W)

    def forward_deduce_states(self, input_statuses, status, deduce_order):
        assert len(input_statuses) == len(self.inputs)
        l2res_map = {-1: 0, 0: -1, 1: 1, 2: 2, 3: 3}
        r2res_map = {-1: 1, 0: -1, 1: 0, 2: 2, 3: 3}
        res2l_map = {-1: 0, 0: -1, 1: 1, 2: 2, 3: 3}
        res2r_map = {-1: 0, 0: 1, 1: -1, 2: 2, 3: 3}
        if deduce_order:
            if input_statuses[0].valid_all():
                order = input_statuses[0].order
                status.set_order(tuple(l2res_map[x] for x in order))
            elif input_statuses[1].valid_all():
                order = input_statuses[1].order
                status.set_order(tuple(r2res_map[x] for x in order))
        else:
            if input_statuses[0].valid_state():
                state, duplicate = input_statuses[0].get()
                keys = dict(state)
                keys[-1] = duplicate
                res_state = tuple(keys.get(res2l_map[x], 1) for x in range(4))
                res_duplicate = keys.get(res2l_map[-1], 1)
                status.set_state(res_state, res_duplicate)
            elif input_statuses[1].valid_state():
                state, duplicate = input_statuses[1].get()
                keys = dict(state)
                keys[-1] = duplicate
                res_state = tuple(keys.get(res2r_map[x], 1) for x in range(4))
                res_duplicate = keys.get(res2r_map[-1], 1)
                status.set_state(res_state, res_duplicate)

    def backward_deduce_states(self, status, input_statuses, deduce_order):
        assert len(input_statuses) == len(self.inputs)
        l2res_map = {-1: 0, 0: -1, 1: 1, 2: 2, 3: 3}
        r2res_map = {-1: 1, 0: -1, 1: 0, 2: 2, 3: 3}
        res2l_map = {-1: 0, 0: -1, 1: 1, 2: 2, 3: 3}
        res2r_map = {-1: 0, 0: 1, 1: -1, 2: 2, 3: 3}
        if deduce_order:
            if status.valid_all():
                res_order = tuple(res2l_map[x] for x in status.order)
                input_statuses[0].set_order(res_order)
                res_order = tuple(res2r_map[x] for x in status.order)
                input_statuses[1].set_order(res_order)
        else:
            if status.valid_state():
                state, duplicate = status.get()
                keys = dict(state)
                keys[-1] = duplicate
                res_state = tuple(keys.get(l2res_map[x], 1) for x in range(4))
                res_duplicate = keys.get(l2res_map[-1], 1)
                input_statuses[0].set_state(res_state, res_duplicate)
                res_state = tuple(keys.get(r2res_map[x], 1) for x in range(4))
                res_duplicate = keys.get(r2res_map[-1], 1)
                input_statuses[1].set_state(res_state, res_duplicate)
            else:
                if input_statuses[0].state is not None:
                    input_statuses[1].set_state(
                        None, input_statuses[0].state.get(1, 1))
                if input_statuses[1].state is not None:
                    input_statuses[0].set_state(
                        None, input_statuses[1].state.get(1, 1))


def conv2d_op(node_A, node_B, padding=0, stride=1, ctx=None):
    """Conv2d node.

    Parameters:
    ----
    node_A : Node
        Input data node.
    node_B : Node
        Input filter node.
    padding :
        Padding size.
    stride :
        Stride size.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return Conv2dOp(node_A, node_B, padding, stride, ctx=ctx)


def conv2d_gradient_of_data_op(node_A, node_B, padding=0, stride=1, ctx=None):
    """Gradient node of data of conv2d.

    Parameters:
    ----
    node_A : Node
        Filter node.
    node_B : Node
        Previous gradient node.
    padding :
        Padding size.
    stride :
        Stride size.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return Conv2d_Gradient_of_DataOp(node_A, node_B, padding, stride, ctx=ctx)


def conv2d_gradient_of_filter_op(input_X, gradient_Y, padding=0, stride=1, ctx=None):
    """Gradient node of filters of conv2d.

    Parameters:
    ----
    input_X :
        Input data of conv2d.
    gradient_Y :
        Gradient array.
    padding :
        Padding size.
    stride :
        Stride size.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return Conv2d_Gradient_of_FilterOp(input_X, gradient_Y, padding, stride, ctx=ctx)
