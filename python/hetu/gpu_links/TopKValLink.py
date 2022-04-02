from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd

def topk_val(in_mat, out_mat_idx, out_mat_val, k, stream=None):
        assert isinstance(in_mat, _nd.NDArray);
        assert isinstance(out_mat_idx, _nd.NDArray);
        assert isinstance(out_mat_val, _nd.NDArray);
                    
        _LIB.DLGpuTopKVal(                            
                in_mat.handle, out_mat_idx.handle, out_mat_val.handle, k, stream.handle if stream else None)
