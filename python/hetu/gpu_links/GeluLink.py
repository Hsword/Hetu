
from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def gelu(in_arr, out_arr, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuGelu(in_arr.handle, out_arr.handle,
                   stream.handle if stream else None)


def gelu_gradient(in_arr, in_grad_arr, out_arr, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(in_grad_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuGeluGradient(in_arr.handle, in_grad_arr.handle,
                           out_arr.handle, stream.handle if stream else None)
