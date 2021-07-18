from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def reduce_mean(in_arr, out_arr, axes, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    pointer_func = ctypes.c_int * len(axes)
    pointer = pointer_func(*list(axes))
    _LIB.DLGpuReduceMean(
        in_arr.handle, out_arr.handle, pointer, ctypes.c_int(len(axes)), stream.handle if stream else None)
