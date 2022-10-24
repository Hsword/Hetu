from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def tanh(in_arr, out_arr, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuTanh(in_arr.handle, out_arr.handle,
                   stream.handle if stream else None)


def tanh_gradient(forward_arr, grad_arr, out_arr, stream=None):
    assert isinstance(forward_arr, _nd.NDArray)
    assert isinstance(grad_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuTanhGradient(forward_arr.handle, grad_arr.handle,
                           out_arr.handle, stream.handle if stream else None)
