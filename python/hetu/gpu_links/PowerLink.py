from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def matrix_power(in_arr, out_arr, p, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuPower(in_arr.handle, out_arr.handle, ctypes.c_float(p),
                    stream.handle if stream else None)
