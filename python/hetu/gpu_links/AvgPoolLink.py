from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def average_pooling2d(in_arr, kernel_H, kernel_W, pooled_layer, padding=0, stride=1, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(pooled_layer, _nd.NDArray)
    _LIB.DLGpuAvgerage_Pooling2d(
        in_arr.handle, kernel_H, kernel_W, pooled_layer.handle, padding, stride, stream.handle if stream else None)


def average_pooling2d_gradient(in_gradient_y, kernel_H, kernel_W, out_gradient_x, padding=0, stride=1, stream=None):
    assert isinstance(in_gradient_y, _nd.NDArray)
    assert isinstance(out_gradient_x, _nd.NDArray)
    _LIB.DLGpuAvgerage_Pooling2d_gradient(
        in_gradient_y.handle, kernel_H, kernel_W, out_gradient_x.handle, padding, stride, stream.handle if stream else None)
