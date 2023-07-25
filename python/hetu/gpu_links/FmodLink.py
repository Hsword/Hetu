
from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def fmod(in_arr, out_arr, val, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuFmod(in_arr.handle, out_arr.handle, ctypes.c_float(
        val), stream.handle if stream else None)
