from __future__ import absolute_import
import numpy as np
from .Node import Op
from .._base import DNNL_LIB
from ..cpu_links import avg_pool as cpu_avg_pool
from ..gpu_links import CuDNN_average_pooling2d
from ..cpu_links import avg_pool_gradient as cpu_avg_pool_gradient
from ..gpu_links import CuDNN_average_pooling2d_gradient


class Avg_Pool2dOp(Op):
    def __init__(self, node_A, kernel_H, kernel_W, padding, stride, ctx=None):
        super().__init__(Avg_Pool2dOp, [node_A], ctx)
        self.padding = padding
        self.stride = stride
        self.kernel_H = kernel_H
        self.kernel_W = kernel_W

    def np_average_pooling(self, input, kernel_H, kernel_W, padding=0, stride=1):
        N, C, H, W = input.shape
        assert((H + 2 * padding - kernel_H) % stride == 0)
        assert((W + 2 * padding - kernel_W) % stride == 0)
        pooled_H = (H + 2 * padding - kernel_H) // stride + 1
        pooled_W = (W + 2 * padding - kernel_W) // stride + 1
        pooled_layer = np.zeros(
            shape=(N, C, pooled_H, pooled_W), dtype=np.float32)
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
                        for i in range(hs, hend):
                            for j in range(ws, wend):
                                pooled_layer[n][c][h][w] += input[n][c][i][j]
                        pooled_layer[n][c][h][w] /= pooling_size
        return pooled_layer

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            if DNNL_LIB['DnnlAvgPool']:
                cpu_avg_pool(
                    input_vals[0], self.kernel_H, self.kernel_W, output_val, self.padding, self.stride)
            else:
                output_val[:] = self.np_average_pooling(
                    input_vals[0].asnumpy(), self.kernel_H, self.kernel_W, self.padding, self.stride)
        else:
            CuDNN_average_pooling2d(
                input_vals[0], self.kernel_H, self.kernel_W, output_val, self.padding, self.stride, stream_handle)

    def gradient(self, output_grad):
        return [avg_pool2d_gradient_op(self, output_grad, self.inputs[0], self.kernel_H, self.kernel_W, self.padding, self.stride, ctx=self.raw_ctx)]

    def infer_shape(self, input_shapes):
        """Need to handle input_vals[0].shape != input_vals[1].shape"""
        assert len(input_shapes) == 1
        N, C, H, W = input_shapes[0]
        p_H = (H + 2 * self.padding - self.kernel_H) // self.stride + 1
        p_W = (W + 2 * self.padding - self.kernel_W) // self.stride + 1
        return (N, C, p_H, p_W)


class Avg_Pool2d_GradientOp(Op):
    def __init__(self, node_out, node_out_gradient, node_in, kernel_H, kernel_W, padding, stride, ctx=None):
        super().__init__(Avg_Pool2d_GradientOp, [
            node_out, node_out_gradient, node_in], ctx)
        self.padding = padding
        self.stride = stride
        self.kernel_H = kernel_H
        self.kernel_W = kernel_W

    def np_average_pooling_gradient(self, gradient_y, kernel_H, kernel_W, padding=0, stride=1):
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
                        for i in range(hs, hend):
                            for j in range(ws, wend):
                                gradient_x[n][c][i][j] += gradient_y[n][c][h][w] / \
                                    pooling_size

        return gradient_x

    def compute(self, input_vals, output_val, stream_handle=None):
        if self.on_cpu:
            if DNNL_LIB['DnnlAvgPool_Gradient']:
                cpu_avg_pool_gradient(
                    input_vals[1], self.kernel_H, self.kernel_W, output_val, self.padding, self.stride)
            else:
                output_val[:] = self.np_average_pooling_gradient(
                    input_vals[1].asnumpy(), self.kernel_H, self.kernel_W, self.padding, self.stride)
        else:
            CuDNN_average_pooling2d_gradient(
                input_vals[0], input_vals[1], input_vals[2], self.kernel_H, self.kernel_W, output_val, self.padding, self.stride, stream_handle)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        assert len(input_shapes) == 3
        return input_shapes[2]


def avg_pool2d_op(node_A, kernel_H, kernel_W, padding, stride, ctx=None):
    """Average pooling node.

    Parameters:
    ----
    node_A : Node
        Input node.
    kernel_H : 
        Kernel height.
    kernel_W :
        Kernel width.
    padding :
        Padding size.
    stride :
        Stride size.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return Avg_Pool2dOp(node_A, kernel_H, kernel_W, padding, stride, ctx=ctx)


def avg_pool2d_gradient_op(node_out, node_out_gradient, node_in, kernel_H, kernel_W, padding, stride, ctx=None):
    """Gradient node of average pooling.

    Parameters:
    ----
    node_out : Node
        Output node of average pooling.
    node_out_gradient : Node
        Previous gradient node.
    node_in : Node
        Input node of average pooling.
    kernel_H : 
        Kernel height.
    kernel_W :
        Kernel width.
    padding :
        Padding size.
    stride :
        Stride size.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return Avg_Pool2d_GradientOp(node_out, node_out_gradient, node_in, kernel_H, kernel_W, padding, stride, ctx=ctx)
