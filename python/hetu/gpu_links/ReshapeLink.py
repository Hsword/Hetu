from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def array_reshape(in_arr, out_arr, stream=None):

    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
#    print("reshape_op_input: ", in_arr.shape)
#    print("reshape_op_output: ", out_arr.shape)
    _LIB.DLGpuReshape(in_arr.handle, out_arr.handle,
                      stream.handle if stream else None)
