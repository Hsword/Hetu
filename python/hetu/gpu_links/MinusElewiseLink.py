from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd

def matrix_elementwise_minus(input1, input2, output, stream=None):
    assert isinstance(input1, _nd.NDArray);
    assert isinstance(input2, _nd.NDArray);
    assert isinstance(output, _nd.NDArray);

    _LIB.DLGpuMinusElewise(input1.handle, input2.handle, output.handle, stream.handle if stream else None)
