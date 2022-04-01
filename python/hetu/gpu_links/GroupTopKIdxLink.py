from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd

def group_topk_idx(in_mat, top1_group, out_mat_idx, k, num_local_gpus, stream=None):
    assert isinstance(in_mat, _nd.NDArray);
    assert isinstance(top1_group, _nd.NDArray);
    assert isinstance(out_mat_idx, _nd.NDArray);                    
    _LIB.DLGpuGroupTopKIdx(                            
        in_mat.handle, top1_group.handle, out_mat_idx.handle, ctypes.c_int(k), ctypes.c_int(num_local_gpus), stream.handle if stream else None)
