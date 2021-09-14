from .base import BaseLayer
import hetu as ht


class MaxPool2d(BaseLayer):
    def __init__(self, kernel_size, stride, padding=0):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def __call__(self, x):
        return ht.max_pool2d_op(
            x, self.kernel_size, self.kernel_size, self.padding, self.stride)


class AvgPool2d(BaseLayer):
    def __init__(self, kernel_size, stride, padding=0):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def __call__(self, x):
        return ht.avg_pool2d_op(
            x, self.kernel_size, self.kernel_size, self.padding, self.stride)
