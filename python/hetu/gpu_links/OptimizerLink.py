from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def add_l2_regularization(param, grad, l2reg, stream=None):
    assert isinstance(param, _nd.NDArray)
    assert isinstance(grad, (_nd.NDArray, _nd.IndexedSlices))
    if isinstance(grad, _nd.NDArray):
        _LIB.AddL2Regularization(param.handle, grad.handle, ctypes.c_float(
            l2reg), stream.handle if stream else None)
    else:
        grad.to_dense(stream)
        assert isinstance(grad.indices, _nd.NDArray)
        assert isinstance(grad.values, _nd.NDArray)
        _LIB.AddL2RegularizationSparse(param.handle, grad.indices.handle, grad.values.handle, ctypes.c_float(
            l2reg), stream.handle if stream else None)


def sgd_update(param, grad, lr, l2reg, stream=None):
    assert isinstance(param, _nd.NDArray)
    assert isinstance(grad, (_nd.NDArray, _nd.IndexedSlices))
    if l2reg > 0:
        add_l2_regularization(param, grad, l2reg, stream)
    if isinstance(grad, _nd.NDArray):
        _LIB.SGDOptimizerUpdate(param.handle, grad.handle, ctypes.c_float(
            lr), stream.handle if stream else None)
    else:
        assert isinstance(grad.indices, _nd.NDArray)
        assert isinstance(grad.values, _nd.NDArray)
        _LIB.SGDOptimizerSparseUpdate(param.handle, grad.indices.handle, grad.values.handle, ctypes.c_float(
            lr), stream.handle if stream else None)
        grad.free_dense()


def momentum_update(param, grad, velocity, lr, momentum, nesterov, l2reg, stream=None):
    assert isinstance(param, _nd.NDArray)
    assert isinstance(grad, (_nd.NDArray, _nd.IndexedSlices))
    assert isinstance(velocity, _nd.NDArray)
    if l2reg > 0:
        add_l2_regularization(param, grad, l2reg, stream)
    if isinstance(grad, _nd.NDArray):
        _LIB.MomentumOptimizerUpdate(param.handle, grad.handle, velocity.handle, ctypes.c_float(
            lr), ctypes.c_float(momentum), ctypes.c_bool(nesterov), stream.handle if stream else None)
    else:
        assert isinstance(grad.indices, _nd.NDArray)
        assert isinstance(grad.values, _nd.NDArray)
        _LIB.MomentumOptimizerSparseUpdate(param.handle, grad.indices.handle, grad.values.handle, velocity.handle, ctypes.c_float(
            lr), ctypes.c_float(momentum), ctypes.c_bool(nesterov), stream.handle if stream else None)
        grad.free_dense()


def adagrad_update(param, grad, accumulation, lr, eps, l2reg, stream=None):
    assert isinstance(param, _nd.NDArray)
    assert isinstance(grad, (_nd.NDArray, _nd.IndexedSlices))
    assert isinstance(accumulation, _nd.NDArray)
    if l2reg > 0:
        add_l2_regularization(param, grad, l2reg, stream)
    if isinstance(grad, _nd.NDArray):
        _LIB.AdaGradOptimizerUpdate(param.handle, grad.handle, accumulation.handle, ctypes.c_float(
            lr), ctypes.c_float(eps), stream.handle if stream else None)
    else:
        if l2reg <= 0:
            grad.deduplicate(stream)
        assert isinstance(grad.indices, _nd.NDArray)
        assert isinstance(grad.values, _nd.NDArray)
        _LIB.AdaGradOptimizerSparseUpdate(param.handle, grad.indices.handle, grad.values.handle, accumulation.handle, ctypes.c_float(
            lr), ctypes.c_float(eps), stream.handle if stream else None)
        if l2reg <= 0:
            grad.free_deduplicate()
        grad.free_dense()


def adam_update(param, grad, expavg, expavgsq, lr, beta1, beta2, beta1t, beta2t, eps, l2reg, stream=None):
    assert isinstance(param, _nd.NDArray)
    assert isinstance(grad, (_nd.NDArray, _nd.IndexedSlices))
    assert isinstance(expavg, _nd.NDArray)
    assert isinstance(expavgsq, _nd.NDArray)
    if l2reg > 0:
        add_l2_regularization(param, grad, l2reg, stream)
    if isinstance(grad, _nd.NDArray):
        _LIB.AdamOptimizerUpdate(param.handle, grad.handle, expavg.handle, expavgsq.handle, ctypes.c_float(lr), ctypes.c_float(beta1), ctypes.c_float(beta2),
                                 ctypes.c_float(beta1t), ctypes.c_float(beta2t), ctypes.c_float(eps), stream.handle if stream else None)
    else:
        if l2reg <= 0:
            grad.deduplicate(stream)
        assert isinstance(grad.indices, _nd.NDArray)
        assert isinstance(grad.values, _nd.NDArray)
        _LIB.AdamOptimizerSparseUpdate(param.handle, grad.indices.handle, grad.values.handle, expavg.handle, expavgsq.handle, ctypes.c_float(lr), ctypes.c_float(beta1), ctypes.c_float(beta2),
                                       ctypes.c_float(beta1t), ctypes.c_float(beta2t), ctypes.c_float(eps), stream.handle if stream else None)
        if l2reg <= 0:
            grad.free_deduplicate()
        grad.free_dense()

def adamw_update(param, grad, expavg, expavgsq, lr, beta1, beta2, beta1t, beta2t, eps, weight_decay, stream=None):
    assert isinstance(param, _nd.NDArray)
    assert isinstance(grad, (_nd.NDArray, _nd.IndexedSlices))
    assert isinstance(expavg, _nd.NDArray)
    assert isinstance(expavgsq, _nd.NDArray)
    if isinstance(grad, _nd.NDArray):
        _LIB.AdamWOptimizerUpdate(param.handle, grad.handle, expavg.handle, expavgsq.handle, ctypes.c_float(lr), ctypes.c_float(beta1), ctypes.c_float(beta2),
                                 ctypes.c_float(beta1t), ctypes.c_float(beta2t), ctypes.c_float(eps), ctypes.c_float(weight_decay), stream.handle if stream else None)
    else:
        grad.deduplicate(stream)
        assert isinstance(grad.indices, _nd.NDArray)
        assert isinstance(grad.values, _nd.NDArray)
        _LIB.AdamWOptimizerSparseUpdate(param.handle, grad.indices.handle, grad.values.handle, expavg.handle, expavgsq.handle, ctypes.c_float(lr), ctypes.c_float(beta1), ctypes.c_float(beta2),
                                       ctypes.c_float(beta1t), ctypes.c_float(beta2t), ctypes.c_float(eps), ctypes.c_float(weight_decay), stream.handle if stream else None)
        grad.free_deduplicate()

def lamb_update(param, grad, expavg, expavgsq, lr, beta1, beta2, beta1t, beta2t, eps, weight_decay, stream=None):
    assert isinstance(param, _nd.NDArray)
    assert isinstance(grad, (_nd.NDArray, _nd.IndexedSlices))
    assert isinstance(expavg, _nd.NDArray)
    assert isinstance(expavgsq, _nd.NDArray)
    if isinstance(grad, _nd.NDArray):
        _LIB.LambOptimizerUpdate(param.handle, grad.handle, expavg.handle, expavgsq.handle, ctypes.c_float(lr), ctypes.c_float(beta1), ctypes.c_float(beta2),
                                 ctypes.c_float(beta1t), ctypes.c_float(beta2t), ctypes.c_float(eps), ctypes.c_float(weight_decay), stream.handle if stream else None)
    else:
        grad.deduplicate(stream)
        assert isinstance(grad.indices, _nd.NDArray)
        assert isinstance(grad.values, _nd.NDArray)
        _LIB.LambOptimizerSparseUpdate(param.handle, grad.indices.handle, grad.values.handle, expavg.handle, expavgsq.handle, ctypes.c_float(lr), ctypes.c_float(beta1), ctypes.c_float(beta2),
                                       ctypes.c_float(beta1t), ctypes.c_float(beta2t), ctypes.c_float(eps), ctypes.c_float(weight_decay), stream.handle if stream else None)
        grad.free_deduplicate()