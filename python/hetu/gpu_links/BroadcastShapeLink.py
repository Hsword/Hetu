from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def broadcast_shape(in_arr, out_arr, add_axes=None, stream=None):
    # deprecated, only used in tests
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    if add_axes is not None:
        pointer_func = ctypes.c_int * len(add_axes)
        pointer = pointer_func(*list(add_axes))
    _LIB.DLGpuBroadcastShape(in_arr.handle, out_arr.handle,
                             pointer if add_axes else None, stream.handle if stream else None)


def broadcast_shape_simple(in_arr, out_arr, out_strides, in_dims, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    assert isinstance(out_strides, _nd.NDArray)
    assert isinstance(in_dims, _nd.NDArray)
    _LIB.DLGpuBroadcastShapeSimple(
        in_arr.handle, out_arr.handle, out_strides.handle, in_dims.handle, stream.handle if stream else None)
