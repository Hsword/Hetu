from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def normal_init(arr, mean, stddev, seed, stream=None):
    assert isinstance(arr, _nd.NDArray)
    _LIB.DLGpuNormalInit(arr.handle, ctypes.c_float(mean), ctypes.c_float(
        stddev), ctypes.c_ulonglong(seed), stream.handle if stream else None)


def uniform_init(arr, lb, ub, seed, stream=None):
    assert isinstance(arr, _nd.NDArray)
    _LIB.DLGpuUniformInit(arr.handle, ctypes.c_float(lb), ctypes.c_float(
        ub), ctypes.c_ulonglong(seed), stream.handle if stream else None)


def truncated_normal_init(arr, mean, stddev, seed, stream=None):
    # time consuming !!
    assert isinstance(arr, _nd.NDArray)
    _LIB.DLGpuTruncatedNormalInit(arr.handle, ctypes.c_float(mean), ctypes.c_float(
        stddev), ctypes.c_ulonglong(seed), stream.handle if stream else None)
