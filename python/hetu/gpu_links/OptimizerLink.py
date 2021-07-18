from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def add_l2_regularization(param, grad, l2reg, stream=None):
    assert isinstance(param, _nd.NDArray)
    assert isinstance(grad, (_nd.NDArray, _nd.IndexedSlices))
    # not support indexed slices now
    if isinstance(grad, _nd.NDArray):
        _LIB.AddL2Regularization(param.handle, grad.handle, ctypes.c_float(
            l2reg), stream.handle if stream else None)


def sgd_update(param, grad, lr, stream=None):
    assert isinstance(param, _nd.NDArray)
    assert isinstance(grad, (_nd.NDArray, _nd.IndexedSlices))
    if isinstance(grad, _nd.NDArray):
        _LIB.SGDOptimizerUpdate(param.handle, grad.handle, ctypes.c_float(
            lr), stream.handle if stream else None)
    else:
        assert isinstance(grad.indices, _nd.NDArray)
        assert isinstance(grad.values, _nd.NDArray)
        _LIB.SGDOptimizerSparseUpdate(param.handle, grad.indices.handle, grad.values.handle, ctypes.c_float(
            lr), stream.handle if stream else None)


def momentum_update(param, grad, velocity, lr, momentum, nesterov, stream=None):
    assert isinstance(param, _nd.NDArray)
    assert isinstance(grad, (_nd.NDArray, _nd.IndexedSlices))
    assert isinstance(velocity, _nd.NDArray)
    if isinstance(grad, _nd.NDArray):
        _LIB.MomentumOptimizerUpdate(param.handle, grad.handle, velocity.handle, ctypes.c_float(
            lr), ctypes.c_float(momentum), ctypes.c_bool(nesterov), stream.handle if stream else None)
    else:
        assert isinstance(grad.indices, _nd.NDArray)
        assert isinstance(grad.values, _nd.NDArray)
        _LIB.MomentumOptimizerSparseUpdate(param.handle, grad.indices.handle, grad.values.handle, velocity.handle, ctypes.c_float(
            lr), ctypes.c_float(momentum), ctypes.c_bool(nesterov), stream.handle if stream else None)


def adagrad_update(param, grad, accumulation, lr, eps, stream=None):
    assert isinstance(param, _nd.NDArray)
    assert isinstance(grad, (_nd.NDArray, _nd.IndexedSlices))
    assert isinstance(accumulation, _nd.NDArray)
    if isinstance(grad, _nd.NDArray):
        _LIB.AdaGradOptimizerUpdate(param.handle, grad.handle, accumulation.handle, ctypes.c_float(
            lr), ctypes.c_float(eps), stream.handle if stream else None)
    else:
        grad.deduplicate(stream)
        assert isinstance(grad.indices, _nd.NDArray)
        assert isinstance(grad.values, _nd.NDArray)
        _LIB.AdaGradOptimizerSparseUpdate(param.handle, grad.indices.handle, grad.values.handle, accumulation.handle, ctypes.c_float(
            lr), ctypes.c_float(eps), stream.handle if stream else None)
        grad.free_deduplicate()


def adam_update(param, grad, expavg, expavgsq, lr, beta1, beta2, beta1t, beta2t, eps, stream=None):
    assert isinstance(param, _nd.NDArray)
    assert isinstance(grad, (_nd.NDArray, _nd.IndexedSlices))
    assert isinstance(expavg, _nd.NDArray)
    assert isinstance(expavgsq, _nd.NDArray)
    if isinstance(grad, _nd.NDArray):
        _LIB.AdamOptimizerUpdate(param.handle, grad.handle, expavg.handle, expavgsq.handle, ctypes.c_float(lr), ctypes.c_float(beta1), ctypes.c_float(beta2),
                                 ctypes.c_float(beta1t), ctypes.c_float(beta2t), ctypes.c_float(eps), stream.handle if stream else None)
    else:
        grad.deduplicate(stream)
        assert isinstance(grad.indices, _nd.NDArray)
        assert isinstance(grad.values, _nd.NDArray)
        _LIB.AdamOptimizerSparseUpdate(param.handle, grad.indices.handle, grad.values.handle, expavg.handle, expavgsq.handle, ctypes.c_float(lr), ctypes.c_float(beta1), ctypes.c_float(beta2),
                                       ctypes.c_float(beta1t), ctypes.c_float(beta2t), ctypes.c_float(eps), stream.handle if stream else None)
        grad.free_deduplicate()
