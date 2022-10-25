from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def pow_matrix(in_arr, out_arr, eps, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuPow(in_arr.handle, out_arr.handle, ctypes.c_float(eps),
                  stream.handle if stream else None)


def pow_gradient(in_arr, in_grad_arr, out_arr, eps, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(in_grad_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuPowGradient(in_arr.handle, in_grad_arr.handle, out_arr.handle,
                          ctypes.c_float(eps), stream.handle if stream else None)
