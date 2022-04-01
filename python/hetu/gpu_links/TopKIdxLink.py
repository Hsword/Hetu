from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd

def topk_idx(in_mat, out_mat_idx, k, stream=None):
    assert isinstance(in_mat, _nd.NDArray);
    assert isinstance(out_mat_idx, _nd.NDArray);                    
    _LIB.DLGpuTopKIdx(                            
        in_mat.handle, out_mat_idx.handle, ctypes.c_int(k), stream.handle if stream else None)
