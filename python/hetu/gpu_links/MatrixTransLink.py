from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def matrix_transpose(in_mat, out_mat, perm, stream=None):
    assert isinstance(in_mat, _nd.NDArray)
    assert isinstance(out_mat, _nd.NDArray)
    
    pointer_func = ctypes.c_int * len(perm)
    perm_ptr = pointer_func(*list(perm))
    pointer_func = ctypes.c_int64 * len(in_mat.shape)
    src_dim_ptr = pointer_func(*list(in_mat.shape))  
    
    _LIB.DLGpuTranspose(in_mat.handle, out_mat.handle,
                       perm_ptr, src_dim_ptr, stream.handle if stream else None)



