from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def roll(input_mat, output_mat, shift, axis, stream=None):
    assert isinstance(input_mat, _nd.NDArray)
    assert isinstance(output_mat, _nd.NDArray)

    nums = len(shift)
    shift_func = ctypes.c_int * len(shift)
    pointer_shift = shift_func(*list(shift))

    if (axis):
        axis_func = ctypes.c_int * len(axis)
        pointer_axis = axis_func(*list(axis))
    else:
        pointer_axis = None

    _LIB.DLGpuRoll(input_mat.handle, pointer_shift, pointer_axis,
                   nums, output_mat.handle, stream.handle if stream else None)
