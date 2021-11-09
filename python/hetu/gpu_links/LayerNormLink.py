from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def layer_normalization(in_arr, ln_scale, ln_bias, mean, var, out_arr, eps, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(ln_scale, _nd.NDArray)
    assert isinstance(ln_bias, _nd.NDArray)
    assert isinstance(mean, _nd.NDArray)
    assert isinstance(var, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuLayerNormalization(in_arr.handle, ln_scale.handle, ln_bias.handle, mean.handle,
                                 var.handle, out_arr.handle, ctypes.c_float(eps), stream.handle if stream else None)


def layer_normalization_gradient(out_grads, in_arr, ln_scale, grad_arr, grad_scale, grad_bias,
                                 mean_arr, var_arr, eps, stream=None):
    assert isinstance(out_grads, _nd.NDArray)
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(ln_scale, _nd.NDArray)
    assert isinstance(grad_arr, _nd.NDArray)
    assert isinstance(grad_scale, _nd.NDArray)
    assert isinstance(grad_bias, _nd.NDArray)
    assert isinstance(mean_arr, _nd.NDArray)
    assert isinstance(var_arr, _nd.NDArray)
    _LIB.DLGpuLayerNormalizationGradient(out_grads.handle, in_arr.handle, ln_scale.handle,
                                         grad_arr.handle, grad_scale.handle, grad_bias.handle,
                                         mean_arr.handle, var_arr.handle, ctypes.c_float(
                                             eps),
                                         stream.handle if stream else None)
