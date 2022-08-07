from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def num_less_than(in_arr, mid_arr, out_arr, threshold, axes, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(mid_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    pointer_func = ctypes.c_int * len(axes)
    pointer = pointer_func(*list(axes))
    _LIB.DLGpuNumLessThan(in_arr.handle, mid_arr.handle, out_arr.handle, ctypes.c_float(
        threshold), pointer, ctypes.c_int(len(axes)), stream.handle if stream else None)


def set_less_than(arr, threshold, stream=None):
    assert isinstance(arr, _nd.NDArray)
    _LIB.DLGpuSetLessThan(arr.handle, ctypes.c_float(threshold),
                          stream.handle if stream else None)
