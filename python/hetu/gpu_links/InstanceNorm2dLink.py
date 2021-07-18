from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def instance_normalization2d(in_arr, mean, var, out_arr, eps, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(mean, _nd.NDArray)
    assert isinstance(var, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuInstanceNormalization2d(in_arr.handle, mean.handle,
                                      var.handle, out_arr.handle, ctypes.c_float(eps), stream.handle if stream else None)


def instance_normalization2d_gradient(out_grads, in_arr, grad_arr, mean_arr, var_arr, eps, stream=None):
    assert isinstance(out_grads, _nd.NDArray)
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(grad_arr, _nd.NDArray)
    assert isinstance(mean_arr, _nd.NDArray)
    assert isinstance(var_arr, _nd.NDArray)
    _LIB.DLGpuInstanceNormalization2dGradient(out_grads.handle, in_arr.handle, grad_arr.handle,
                                              mean_arr.handle, var_arr.handle, ctypes.c_float(
                                                  eps),
                                              stream.handle if stream else None)
