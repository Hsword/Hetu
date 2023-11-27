from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def arange(start, end, step, out_mat, stream=None):
    assert isinstance(out_mat, _nd.NDArray)
    _LIB.DLGpuArange(ctypes.c_float(start), ctypes.c_float(end), ctypes.c_float(
        step), out_mat.handle, stream.handle if stream else None)
