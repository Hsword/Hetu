from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def matrix_slice(in_arr, out_arr, begin_pos, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    pointer_func = ctypes.c_int64 * len(begin_pos)
    pointer = pointer_func(*list(begin_pos))
    _LIB.DLGpuSlice(in_arr.handle, out_arr.handle, pointer,
                    stream.handle if stream else None)


def matrix_slice_gradient(in_arr, out_arr, begin_pos, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    pointer_func = ctypes.c_int64 * len(begin_pos)
    pointer = pointer_func(*list(begin_pos))
    _LIB.DLGpuSliceGradient(in_arr.handle, out_arr.handle,
                            pointer, stream.handle if stream else None)


def matrix_slice_simple(in_arr, out_arr, gpu_buf, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    assert isinstance(gpu_buf, _nd.NDArray)
    _LIB.DLGpuSliceSimple(in_arr.handle, out_arr.handle,
                          gpu_buf.handle, stream.handle if stream else None)


def matrix_slice_gradient_simple(in_arr, out_arr, gpu_buf, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    assert isinstance(gpu_buf, _nd.NDArray)
    _LIB.DLGpuSliceGradientSimple(
        in_arr.handle, out_arr.handle, gpu_buf.handle, stream.handle if stream else None)
