from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def normal_init(arr, mean, stddev, stream=None):
    assert isinstance(arr, _nd.NDArray)
    _LIB.DLGpuNormalInit(arr.handle, ctypes.c_float(
        mean), ctypes.c_float(stddev), stream.handle if stream else None)


def uniform_init(arr, lb, ub, stream=None):
    assert isinstance(arr, _nd.NDArray)
    _LIB.DLGpuUniformInit(arr.handle, ctypes.c_float(lb), ctypes.c_float(
        ub), stream.handle if stream else None)


def truncated_normal_init(arr, mean, stddev, stream=None):
    # time consuming !!
    assert isinstance(arr, _nd.NDArray)
    _LIB.DLGpuTruncatedNormalInit(arr.handle, ctypes.c_float(mean), ctypes.c_float(
        stddev), stream.handle if stream else None)


def reversed_truncated_normal_init(arr, mean, stddev, stream=None):
    # time consuming !!
    assert isinstance(arr, _nd.NDArray)
    _LIB.DLGpuReversedTruncatedNormalInit(arr.handle, ctypes.c_float(mean), ctypes.c_float(
        stddev), stream.handle if stream else None)


def gumbel_init(arr, stream=None):
    assert isinstance(arr, _nd.NDArray)
    _LIB.DLGpuGumbelInit(arr.handle, stream.handle if stream else None)


def randint_init(arr, lb, ub, stream=None):
    assert isinstance(arr, _nd.NDArray)
    _LIB.DLGpuRandomInt(arr.handle, ctypes.c_int(lb), ctypes.c_int(
        ub), stream.handle if stream else None)
