from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def norm(in_arr, out_arr, axis, p, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuNorm(in_arr.handle, out_arr.handle, ctypes.c_int(axis), ctypes.c_int(p),
                   stream.handle if stream else None)


def norm_gradient(input, input_y, grad_y, output, axis, p, stream=None):
    assert isinstance(input, _nd.NDArray)
    assert isinstance(input_y, _nd.NDArray)
    assert isinstance(grad_y, _nd.NDArray)
    assert isinstance(output, _nd.NDArray)
    _LIB.DLGpuNormGradient(input.handle, input_y.handle, grad_y.handle, output.handle, ctypes.c_int(axis), ctypes.c_int(p),
                           stream.handle if stream else None)
