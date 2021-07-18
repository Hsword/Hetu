from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def array_set(arr, value, stream=None):
    assert isinstance(arr, _nd.NDArray)
    _LIB.DLGpuArraySet(arr.handle, ctypes.c_float(
        value), stream.handle if stream else None)
