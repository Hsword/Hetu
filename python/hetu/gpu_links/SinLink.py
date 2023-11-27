from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def sin(in_arr, out_arr, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuSin(in_arr.handle, out_arr.handle,
                  stream.handle if stream else None)


def cos(in_arr, out_arr, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuCos(in_arr.handle, out_arr.handle,
                  stream.handle if stream else None)
