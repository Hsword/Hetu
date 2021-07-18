from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def indexedslice_oneside_add(indslice, output, stream=None):
    assert isinstance(indslice.indices, _nd.NDArray)
    assert isinstance(indslice.values, _nd.NDArray)
    assert isinstance(output, _nd.NDArray)
    _LIB.IndexedSlicesOneSideAdd(indslice.indices.handle, indslice.values.handle, output.handle,
                                 stream.handle if stream else None)
