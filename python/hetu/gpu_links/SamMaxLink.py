from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd

def sammax_link(in_mat, top1_group, topk_indice, out_mat, num_local_gpus, stream=None):
    assert isinstance(in_mat, _nd.NDArray)
    assert isinstance(top1_group, _nd.NDArray)
    assert isinstance(topk_indice, _nd.NDArray)
    assert isinstance(out_mat, _nd.NDArray)                    
    _LIB.DLGpuSamMax(                            
        in_mat.handle, top1_group.handle, topk_indice.handle, out_mat.handle, ctypes.c_int(num_local_gpus), stream.handle if stream else None)



def sammax_grad_link(output_grad, in_mat, top1_group, topk_indice, out_mat, num_local_gpus, stream=None):
    assert isinstance(output_grad, _nd.NDArray)
    assert isinstance(in_mat, _nd.NDArray)
    assert isinstance(top1_group, _nd.NDArray)
    assert isinstance(topk_indice, _nd.NDArray)
    assert isinstance(out_mat, _nd.NDArray)                    
    _LIB.DLGpuSamMaxGrad(                            
        output_grad.handle, in_mat.handle, top1_group.handle, topk_indice.handle, out_mat.handle, ctypes.c_int(num_local_gpus), stream.handle if stream else None)
