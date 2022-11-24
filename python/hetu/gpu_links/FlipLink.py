from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def flip(in_arr, out_arr, dims, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    pointer_func = ctypes.c_int * len(dims)
    pointer = pointer_func(*list(dims))

    _LIB.DLGpuFlip(in_arr.handle, out_arr.handle, pointer, ctypes.c_int(
        len(dims)), stream.handle if stream else None)
