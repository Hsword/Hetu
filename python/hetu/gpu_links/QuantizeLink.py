from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def tensor_quantize(in_arr, out_arr, digit, scale, minele, seed, stochastic=True, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuRoundingToInt(in_arr.handle, out_arr.handle, ctypes.c_float(scale), ctypes.c_float(
        minele), ctypes.c_int(digit), seed, ctypes.c_bool(stochastic), stream.handle if stream else None)


def tensor_dequantize(in_arr, out_arr, digit, scale, minele, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuDequantize(in_arr.handle, out_arr.handle, ctypes.c_int(digit), ctypes.c_float(
        scale), ctypes.c_float(minele), stream.handle if stream else None)
