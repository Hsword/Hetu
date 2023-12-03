from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def num_less_than(in_arr, mid_arr, out_arr, threshold, axes=None, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(mid_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    if axes is None:
        axes = list(range(len(mid_arr.shape)))
    pointer_func = ctypes.c_int * len(axes)
    pointer = pointer_func(*list(axes))
    _LIB.DLGpuNumLessThan(in_arr.handle, mid_arr.handle, out_arr.handle, ctypes.c_float(
        threshold), pointer, ctypes.c_int(len(axes)), stream.handle if stream else None)


def set_less_than(arr, threshold, stream=None):
    assert isinstance(arr, _nd.NDArray)
    _LIB.DLGpuSetLessThan(arr.handle, ctypes.c_float(threshold),
                          stream.handle if stream else None)


def set_mask_less_than(arr, mask, threshold, stream=None):
    assert isinstance(arr, _nd.NDArray)
    assert isinstance(mask, _nd.NDArray)
    _LIB.DLGpuSetMaskLessThan(arr.handle, mask.handle, ctypes.c_float(
        threshold), stream.handle if stream else None)


def num_less_than_tensor_threshold(in_arr, mid_arr, out_arr, threshold, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(mid_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    assert isinstance(threshold, _nd.NDArray)
    axes = list(range(len(in_arr.shape)))
    pointer_func = ctypes.c_int * len(axes)
    pointer = pointer_func(*axes)
    _LIB.DLGpuNumLessThanTensorThreshold(in_arr.handle, mid_arr.handle, out_arr.handle, threshold.handle,
                                         pointer, ctypes.c_int(len(axes)), stream.handle if stream else None)


def get_larger_than(in_arr, threshold, mask, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(threshold, _nd.NDArray)
    assert isinstance(mask, _nd.NDArray)
    _LIB.DLGpuGetLargerThan(in_arr.handle, threshold.handle,
                            mask.handle, stream.handle if stream else None)


def multiply_grouping_alpha(arr, grouping, alpha, stream=None):
    assert isinstance(arr, _nd.NDArray)
    assert isinstance(grouping, _nd.NDArray)
    assert isinstance(alpha, _nd.NDArray)
    _LIB.DLGpuMultiplyGroupingAlpha(
        arr.handle, grouping.handle, alpha.handle, stream.handle if stream else None)
