from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def dropout(in_arr, dropout_rate, out_arr, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuDropout(in_arr.handle, ctypes.c_float(
        dropout_rate), out_arr.handle, stream.handle if stream else None)


def dropout_gradient_recompute(grad_arr, dropout_rate, out_arr, seed_seqnum, stream=None):
    assert isinstance(grad_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuDropoutGradient_recompute(grad_arr.handle, ctypes.c_float(
        dropout_rate), out_arr.handle, seed_seqnum, stream.handle if stream else None)


def dropout_gradient(grad_arr, fw_output_arr, dropout_rate, out_arr, stream=None):
    assert isinstance(grad_arr, _nd.NDArray)
    assert isinstance(fw_output_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuDropoutGradient(grad_arr.handle, fw_output_arr.handle, ctypes.c_float(
        dropout_rate), out_arr.handle, stream.handle if stream else None)
