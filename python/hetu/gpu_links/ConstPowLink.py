from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def const_pow(in_arr, out_arr, val, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuConstPow(in_arr.handle, ctypes.c_float(val),
                       out_arr.handle, stream.handle if stream else None)


def const_pow_gradient(in_arr, grad_arr, out_arr, val, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(grad_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuConstPowGradient(
        in_arr.handle, grad_arr.handle, ctypes.c_float(val), out_arr.handle, stream.handle if stream else None)
