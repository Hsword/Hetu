from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd

def log_link(input, output, stream=None):
    assert isinstance(input, _nd.NDArray)
    assert isinstance(output, _nd.NDArray)
#    import numpy as np
#    import hetu as ht
#    if input.ctx==ht.gpu(0):
#        np.savetxt("softmax_result2.txt", input.asnumpy())
    _LIB.DLGpuLog(input.handle, output.handle, stream.handle if stream else None)

def log_grad_link(output_grad, input, input_grad, stream=None):
    assert isinstance(output_grad, _nd.NDArray)
    assert isinstance(input, _nd.NDArray)
    assert isinstance(input_grad, _nd.NDArray)

#    import numpy as np
#    import hetu as ht
#    if output_grad.ctx==ht.gpu(0):
#        np.savetxt("log_output_grad2.txt", output_grad.asnumpy())

    _LIB.DLGpuLogGrad(output_grad.handle, input.handle, input_grad.handle, stream.handle if stream else None)
