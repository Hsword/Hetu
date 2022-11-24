from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def triu(in_arr, out_arr, diagonal, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuTriuTril(in_arr.handle, out_arr.handle, False, ctypes.c_int(
        diagonal), stream.handle if stream else None)


def tril(in_arr, out_arr, diagonal, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuTriuTril(in_arr.handle, out_arr.handle, True, ctypes.c_int(
        diagonal), stream.handle if stream else None)
