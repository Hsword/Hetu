from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def param_clip_func(arr, min_value, max_value, stream=None):
    assert isinstance(arr, _nd.NDArray)
    _LIB.DLGpuClipping(arr.handle, ctypes.c_float(min_value), ctypes.c_float(
        max_value), stream.handle if stream else None)
