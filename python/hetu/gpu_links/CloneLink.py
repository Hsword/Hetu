from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd

def clone(input_mat, output_mat, stream=None):
        assert isinstance(input_mat, _nd.NDArray);
        assert isinstance(output_mat, _nd.NDArray);
                    
        _LIB.DLGpuClone(                       
                input_mat.handle, output_mat.handle, stream.handle if stream else None)
