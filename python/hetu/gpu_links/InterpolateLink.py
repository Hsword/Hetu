from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def bicubic_interpolate(in_arr, out_arr, align_corners, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuBicubicInterpolate(
        in_arr.handle, out_arr.handle, align_corners, stream.handle if stream else None)


def bicubic_interpolate_gradient(input_grad, output, align_corners, stream=None):
    assert isinstance(input_grad, _nd.NDArray)
    assert isinstance(output, _nd.NDArray)
    _LIB.DLGpuBicubicInterpolateGradient(
        output.handle, input_grad.handle, align_corners, stream.handle if stream else None)
