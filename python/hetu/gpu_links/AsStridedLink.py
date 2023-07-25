from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def as_strided(in_arr, out_arr, stride, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    pointer_func = ctypes.c_int * len(stride)
    pointer = pointer_func(*list(stride))

    _LIB.DLGpuAsStrided(in_arr.handle, out_arr.handle,
                        pointer, stream.handle if stream else None)


def as_strided_gradient(grad_arr, out_arr, stride, stream=None):
    assert isinstance(grad_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    pointer_func = ctypes.c_int * len(stride)
    pointer = pointer_func(*list(stride))

    _LIB.DLGpuAsStridedGradient(
        grad_arr.handle, out_arr.handle, pointer, stream.handle if stream else None)
