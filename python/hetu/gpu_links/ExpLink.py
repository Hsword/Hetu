from __future__ import absolute_import

from .._base import _LIB
from .. import ndarray as _nd


def exp_func(input, output, stream=None):
    assert isinstance(input, _nd.NDArray)
    assert isinstance(output, _nd.NDArray)
    _LIB.DLGpuExp(input.handle, output.handle,
                  stream.handle if stream else None)
