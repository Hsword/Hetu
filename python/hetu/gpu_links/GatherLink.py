
from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def gather(in_arr, index, out_arr, dim, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(index, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuGather(in_arr.handle, index.handle, out_arr.handle,
                     ctypes.c_int(dim), stream.handle if stream else None)


def gather_gradient(in_arr, index, out_arr, dim, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(index, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuGatherGradient(in_arr.handle, index.handle, out_arr.handle, ctypes.c_int(
        dim), stream.handle if stream else None)
