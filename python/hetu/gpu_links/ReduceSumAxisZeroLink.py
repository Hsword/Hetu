from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def reduce_sum_axis_zero(in_arr, out_arr, workspace_arr, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DLGpuReduceSumAxisZero(
        in_arr.handle, out_arr.handle, stream.handle if stream else None)


def _reduce_sum_axis_zero(in_arr, out_arr, workspace_arr, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    assert isinstance(workspace_arr, _nd.NDArray)
    _LIB._DLGpuReduceSumAxisZero(
        in_arr.handle, out_arr.handle, workspace_arr.handle, stream.handle if stream else None)
