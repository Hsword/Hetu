from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def slice_assign(in_arr, out_arr, val, begin_pos, end_pos, stream=None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    pointer_func = ctypes.c_int * len(begin_pos)
    pointer_begin = pointer_func(*list(begin_pos))
    pointer_end = pointer_func(*list(end_pos))
    _LIB.DLGpuSliceAssign(in_arr.handle, out_arr.handle, ctypes.c_float(
        val), pointer_begin, pointer_end, stream.handle if stream else None)


def slice_assign_matrix(arr_A, arr_B, out_arr, begin_pos_A, end_pos_A, begin_pos_B, stream=None):
    assert isinstance(arr_A, _nd.NDArray)
    assert isinstance(arr_B, _nd.NDArray)
    pointer_func = ctypes.c_int * len(begin_pos_A)
    pointer_begin_A = pointer_func(*list(begin_pos_A))
    pointer_end_A = pointer_func(*list(end_pos_A))
    pointer_begin_B = pointer_func(*list(begin_pos_B))

    _LIB.DLGpuSliceAssignMatrix(arr_A.handle, arr_B.handle, out_arr.handle, pointer_begin_A,
                                pointer_end_A, pointer_begin_B, stream.handle if stream else None)
