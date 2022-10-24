from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def clamp(input_mat, output_mat, mmin=None, mmax=None, min_mat=None, max_mat=None, stream=None):
    assert isinstance(input_mat, _nd.NDArray)
    assert isinstance(output_mat, _nd.NDArray)

    if mmin == None and mmax != None and min_mat == None and max_mat == None:
        _LIB.DLGpuClampMax(input_mat.handle, ctypes.c_float(
            mmax), output_mat.handle, stream.handle if stream else None)
    elif mmin != None and mmax == None and min_mat == None and max_mat == None:
        _LIB.DLGpuClampMin(input_mat.handle, ctypes.c_float(
            mmin), output_mat.handle, stream.handle if stream else None)
    elif mmin != None and mmax != None and min_mat == None and max_mat == None:
        _LIB.DLGpuClamp(input_mat.handle, ctypes.c_float(mmin), ctypes.c_float(
            mmax), output_mat.handle, stream.handle if stream else None)
    elif mmin == None and mmax == None:
        _LIB.DLGpuClampMat(input_mat.handle, min_mat.handle if min_mat else None,
                           max_mat.handle if max_mat else None, output_mat.handle, stream.handle if stream else None)
    else:
        assert False
