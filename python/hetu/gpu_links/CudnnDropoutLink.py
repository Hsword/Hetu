from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def CuDNN_Dropout(in_arr, dropout, out_arr, reserve_size, reserve_space, firstTime, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.CuDNN_DLGpuDropout(in_arr.handle, ctypes.c_float(dropout), out_arr.handle, ctypes.byref(reserve_size),
                            ctypes.byref(reserve_space), ctypes.c_int(firstTime), stream.handle if stream else None)


def CuDNN_Dropout_gradient(in_gradient_y, dropout, out_gradient_x, reserve_size, reserve_space, stream=None):
    assert isinstance(in_gradient_y, _nd.NDArray)
    assert isinstance(out_gradient_x, _nd.NDArray)
    _LIB.CuDNN_DLGpuDropout_gradient(
        in_gradient_y.handle, ctypes.c_float(
            dropout), out_gradient_x.handle, ctypes.byref(reserve_size),
        ctypes.byref(reserve_space), stream.handle if stream else None)
