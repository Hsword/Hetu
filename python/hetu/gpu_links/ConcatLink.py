from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def concat(in_arr1, in_arr2, out_arr, axis=0, stream=None):
    assert isinstance(in_arr1, _nd.NDArray)
    assert isinstance(in_arr2, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuConcat(in_arr1.handle, in_arr2.handle,
                     out_arr.handle, axis, stream.handle if stream else None)


def concat_gradient(out_grad_arr, in_arr, axis=0, idx=0, stream=None):
    assert isinstance(out_grad_arr, _nd.NDArray)
    assert isinstance(in_arr, _nd.NDArray)
    _LIB.DLGpuConcat_gradient(
        out_grad_arr.handle, in_arr.handle, axis, idx, stream.handle if stream else None)
