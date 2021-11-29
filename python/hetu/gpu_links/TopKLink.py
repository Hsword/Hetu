from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd

def topk(in_mat, out_mat_val, out_mat_idx, k, stream=None):
        assert isinstance(in_mat, _nd.NDArray);
        assert isinstance(out_mat_val, _nd.NDArray);
        assert isinstance(out_mat_idx, _nd.NDArray);
                    
        _LIB.DLGpuTopK(                            
                in_mat.handle, out_mat_val.handle, out_mat_idx.handle, k, stream.handle if stream else None)
