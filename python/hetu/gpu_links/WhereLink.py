from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def where(cond, arr1, arr2, out_arr, stream=None):
    assert isinstance(cond, _nd.NDArray)
    assert isinstance(arr1, _nd.NDArray)
    assert isinstance(arr2, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuWhere(cond.handle, arr1.handle, arr2.handle,
                    out_arr.handle, stream.handle if stream else None)


def where_const(cond, arr1, const_attr, out_arr, stream=None):
    assert isinstance(cond, _nd.NDArray)
    assert isinstance(arr1, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuWhereConst(cond.handle, arr1.handle, ctypes.c_float(const_attr),
                         out_arr.handle, stream.handle if stream else None)
