from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def concatenate(in_arrs, out_arr, axis=0, stream=None):
    assert isinstance(out_arr, _nd.NDArray)
    offset = 0
    for arr in in_arrs:
        assert isinstance(arr, _nd.NDArray)
        _LIB.DLGpuConcatenate(
            arr.handle, out_arr.handle,
            ctypes.c_int(axis), ctypes.c_int(offset),
            stream.handle if stream else None)
        offset += arr.handle.contents.shape[axis]


def concatenate_gradient(out_grad_arr, in_arr, axis, offset, stream=None):
    assert isinstance(out_grad_arr, _nd.NDArray)
    assert isinstance(in_arr, _nd.NDArray)
    _LIB.DLGpuConcatenate_gradient(
        out_grad_arr.handle, in_arr.handle,
        ctypes.c_int(axis), ctypes.c_int(offset),
        stream.handle if stream else None)
