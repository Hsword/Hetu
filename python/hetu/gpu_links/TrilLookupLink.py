from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def tril_lookup(in_arr, out_arr, offset, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuTrilLookup(
        in_arr.handle, out_arr.handle, ctypes.c_int(offset), stream.handle if stream else None)


def tril_lookup_gradient(in_arr, out_arr, offset, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuTrilLookupGradient(
        in_arr.handle, out_arr.handle, ctypes.c_int(offset), stream.handle if stream else None)
