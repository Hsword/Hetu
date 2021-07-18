from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def max_pooling2d(in_arr, kernel_H, kernel_W, pooled_layer, padding=0, stride=1, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(pooled_layer, _nd.NDArray)
    _LIB.DLGpuMax_Pooling2d(in_arr.handle, kernel_H,
                            kernel_W, pooled_layer.handle, padding, stride, stream.handle if stream else None)


def max_pooling2d_gradient(in_arr, in_grad_arr, kernel_H, kernel_W, out_grad_arr, padding=0, stride=1, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(in_grad_arr, _nd.NDArray)
    assert isinstance(out_grad_arr, _nd.NDArray)
    _LIB.DLGpuMax_Pooling2d_gradient(
        in_arr.handle, in_grad_arr.handle, kernel_H, kernel_W, out_grad_arr.handle, padding, stride, stream.handle if stream else None)
