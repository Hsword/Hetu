from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd

def sam_group_sum_link(gate_mat, out_mat, num_local_gpus, stream=None):
    assert isinstance(gate_mat, _nd.NDArray);
    assert isinstance(out_mat, _nd.NDArray);                    
    _LIB.DLGpuSamGroupSum(                            
        gate_mat.handle, out_mat.handle, ctypes.c_int(num_local_gpus), stream.handle if stream else None)
