from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def dropout2d(in_arr, dropout2d_rate, out_arr, seed, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuDropout2d(in_arr.handle, ctypes.c_float(
        dropout2d_rate), out_arr.handle, ctypes.byref(seed), stream.handle if stream else None)


def dropout2d_gradient(grad_arr, dropout2d_rate, out_arr, seed, stream=None):
    assert isinstance(grad_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuDropout2dGradient(grad_arr.handle, ctypes.c_float(
        dropout2d_rate), out_arr.handle, seed, stream.handle if stream else None)
