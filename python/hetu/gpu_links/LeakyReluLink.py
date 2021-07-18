from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def leaky_relu(in_arr, alpha, out_arr, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuLeakyRelu(in_arr.handle, ctypes.c_float(
        alpha), out_arr.handle, stream.handle if stream else None)


def leaky_relu_gradient(in_arr, in_grad_arr, alpha, out_arr, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(in_grad_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuLeakyReluGradient(in_arr.handle, in_grad_arr.handle, ctypes.c_float(
        alpha), out_arr.handle, stream.handle if stream else None)
