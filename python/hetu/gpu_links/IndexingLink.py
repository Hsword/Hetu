from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd

def indexing(input_mat, index_mat,output_mat, stream=None):
    assert isinstance(input_mat, _nd.NDArray);
    assert isinstance(index_mat, _nd.NDArray);
    assert isinstance(output_mat, _nd.NDArray);
    
    _LIB.DLGpuIndexing(
            input_mat.handle,index_mat.handle, output_mat.handle, stream.handle if stream else None)

def indexing_grad(output_grad, index, input_grad, stream=None):
    assert isinstance(output_grad, _nd.NDArray);
    assert isinstance(index, _nd.NDArray);
    assert isinstance(input_grad, _nd.NDArray);

    _LIB.DLGpuIndexingGrad(output_grad.handle, index.handle, input_grad.handle, stream.handle)
