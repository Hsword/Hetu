from __future__ import absolute_import
import numpy as np
from .Node import Op
from .._base import DNNL_LIB
from ..cpu_links import max_pool as cpu_max_pooling
from ..cpu_links import max_pool_gradient as cpu_max_pooling_gradient
from ..gpu_links import CuDNN_max_pooling2d
from ..gpu_links import CuDNN_max_pooling2d_gradient


def np_max_pooling(input, kernel_H, kernel_W, padding=0, stride=1):
    N, C, H, W = input.shape
    assert((H + 2 * padding - kernel_H) % stride == 0)
    assert((W + 2 * padding - kernel_W) % stride == 0)
    pooled_H = (H + 2 * padding - kernel_H) // stride + 1
    pooled_W = (W + 2 * padding - kernel_W) // stride + 1

    pooled_layer = np.zeros(shape=(N, C, pooled_H, pooled_W), dtype=np.float32)
    pooling_size = kernel_H * kernel_W

    for n in range(N):
        for c in range(C):
            for h in range(pooled_H):
                for w in range(pooled_W):
                    hs = h * stride - padding
                    ws = w * stride - padding
                    hend = min(hs + kernel_H, H)
                    wend = min(ws + kernel_W, W)
                    hs = max(hs, 0)
                    ws = max(ws, 0)

                    hargmax = hs
                    wargmax = ws
                    for i in range(hs, hend):
                        for j in range(ws, wend):
                            if input[n][c][i][j] > input[n][c][hargmax][wargmax]:
                                hargmax = i
                                wargmax = j
                    pooled_layer[n][c][h][w] = input[n][c][hargmax][wargmax]

    return pooled_layer


def np_max_pooling_gradient(input, gradient_y, kernel_H, kernel_W, padding=0, stride=1):
    N, C, pooled_H, pooled_W = gradient_y.shape
    H = (pooled_H - 1) * stride + kernel_H - 2 * padding
    W = (pooled_W - 1) * stride + kernel_W - 2 * padding
    gradient_x = np.zeros(shape=(N, C, H, W), dtype=np.float32)
    pooling_size = kernel_H * kernel_W

    for n in range(N):
        for c in range(C):
            for h in range(pooled_H):
                for w in range(pooled_W):
                    hs = h * stride - padding
                    ws = w * stride - padding
                    hend = min(hs + kernel_H, H)
                    wend = min(ws + kernel_W, W)
                    hs = max(hs, 0)
                    ws = max(ws, 0)

                    hargmax = hs
                    wargmax = ws
                    for i in range(hs, hend):
                        for j in range(ws, wend):
                            if input[n][c][i][j] > input[n][c][hargmax][wargmax]:
                                hargmax = i
                                wargmax = j
                    gradient_x[n][c][hargmax][wargmax] += gradient_y[n][c][h][w]

    return gradient_x


class Max_Pool2dOp(Op):
    def __init__(self, node_A, kernel_H, kernel_W, padding, stride, ctx=None):
        super().__init__(Max_Pool2dOp, [node_A], ctx)
        self.padding = padding
        self.stride = stride
        self.kernel_H = kernel_H
        self.kernel_W = kernel_W

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            if DNNL_LIB['DnnlMaxPool']:
                cpu_max_pooling(
                    input_vals[0], self.kernel_H, self.kernel_W, output_val, self.padding, self.stride)
            else:
                output_val[:] = np_max_pooling(input_vals[0].asnumpy(
                ), self.kernel_H, self.kernel_W, self.padding, self.stride)
        else:
            CuDNN_max_pooling2d(
                input_vals[0], self.kernel_H, self.kernel_W, output_val, self.padding, self.stride, stream_handle)

    def gradient(self, output_grad):
        return [max_pool2d_gradient_op(self, output_grad, self.inputs[0], self.kernel_H, self.kernel_W, self.padding, self.stride, ctx=self.raw_ctx)]

    def infer_shape(self, input_shapes):
        """Need to handle input_vals[0].shape != input_vals[1].shape"""
        assert len(input_shapes) == 1
        N, C, H, W = input_shapes[0]
        p_H = (H + 2 * self.padding - self.kernel_H) // self.stride + 1
        p_W = (W + 2 * self.padding - self.kernel_W) // self.stride + 1
        return (N, C, p_H, p_W)


class Max_Pool2d_GradientOp(Op):
    def __init__(self, node_out, node_out_gradient, node_in, kernel_H, kernel_W, padding, stride, ctx=None):
        super().__init__(Max_Pool2d_GradientOp, [
            node_out, node_out_gradient, node_in], ctx)
        self.padding = padding
        self.stride = stride
        self.kernel_H = kernel_H
        self.kernel_W = kernel_W

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            if DNNL_LIB['DnnlMaxPool_Gradient']:
                cpu_max_pooling_gradient(
                    input_vals[2], input_vals[1], self.kernel_H, self.kernel_W, output_val, self.padding, self.stride)
            else:
                output_val[:] = np_max_pooling_gradient(input_vals[2].asnumpy(
                ), input_vals[1].asnumpy(), self.kernel_H, self.kernel_W, self.padding, self.stride)
        else:
            CuDNN_max_pooling2d_gradient(
                input_vals[0], input_vals[1], input_vals[2], self.kernel_H, self.kernel_W, output_val, self.padding, self.stride, stream_handle)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 3
        return input_shapes[2]


def max_pool2d_op(node_A, kernel_H, kernel_W, padding, stride, ctx=None):
    """Make a new instance of Max_Pool2dOp and call the instance.

    Parameters:
    ----
    node_A : Node
        Input Node
    kernel_H : scalar value
        Size of pool(height)
    kernel_W : scalar value
        Size of pool(width)
    padding : scalar value
        Padding edge
    stride : scalar value
        Step Length of the kernel

    Returns:
    ----
    A new Node instance created by Op.

    """
    return Max_Pool2dOp(node_A, kernel_H, kernel_W, padding, stride, ctx=ctx)


def max_pool2d_gradient_op(node_out, node_out_gradient, node_in, kernel_H, kernel_W, padding, stride, ctx=None):
    """Make a new instance of Max_Pool2d_GradientOp and call the instance.

    Parameters:
    ----
    node_out : Node
        Output Node
    node_out_gradient : Node
        Gradient array
    node_in : Node
        Input Node
    kernel_H : scalar value
        Size of pool(height)
    kernel_W : scalar value
        Size of pool(width)
    padding : scalar value
        Padding edge
    stride : scalar value
        Step Length of the kernel

    Returns:
    ----
    A new Node instance created by Op.

    """
    return Max_Pool2d_GradientOp(node_out, node_out_gradient, node_in, kernel_H, kernel_W, padding, stride, ctx=ctx)
