from __future__ import absolute_import

import ctypes
from .._base import _LIB
from ..ndarray import NDArray, IndexedSlices, RobeSlices


def add_l2_regularization(param, grad, l2reg, stream=None):
    if l2reg > 0:
        if isinstance(grad, IndexedSlices):
            grad = grad.to_dense(stream)
        _LIB.AddL2Regularization(param.handle, grad.handle, ctypes.c_float(
            l2reg), stream.handle if stream else None)
    return grad


def sgd_update(param, grad, lr, l2reg, stream=None):
    assert isinstance(param, NDArray)
    assert isinstance(grad, (NDArray, IndexedSlices, RobeSlices))
    grad = add_l2_regularization(param, grad, l2reg, stream)
    if isinstance(grad, RobeSlices):
        assert isinstance(grad.indices, NDArray)
        assert isinstance(grad.values, NDArray)
        assert isinstance(grad.x, NDArray)
        _LIB.SGDOptimizerRobeUpdate(param.handle, grad.indices.handle, grad.values.handle, grad.x.handle, ctypes.c_float(
            lr), ctypes.c_int(grad.Bg), ctypes.c_int(grad.Cg), ctypes.c_int(grad.Dg), stream.handle if stream else None)
    elif isinstance(grad, IndexedSlices):
        grad.deduplicate(stream)
        assert isinstance(grad.dedup_ind, NDArray)
        assert isinstance(grad.dedup_val, NDArray)
        _LIB.SGDOptimizerSparseUpdate(param.handle, grad.dedup_ind.handle, grad.dedup_val.handle, ctypes.c_float(
            lr), stream.handle if stream else None)
    else:
        _LIB.SGDOptimizerUpdate(param.handle, grad.handle, ctypes.c_float(
            lr), stream.handle if stream else None)


def momentum_update(param, grad, velocity, lr, momentum, nesterov, l2reg, stream=None):
    assert isinstance(param, NDArray)
    assert isinstance(grad, (NDArray, IndexedSlices))
    assert isinstance(velocity, NDArray)
    grad = add_l2_regularization(param, grad, l2reg, stream)
    if isinstance(grad, IndexedSlices):
        grad = grad.to_dense()
    assert isinstance(grad, NDArray)
    _LIB.MomentumOptimizerUpdate(param.handle, grad.handle, velocity.handle, ctypes.c_float(
        lr), ctypes.c_float(momentum), ctypes.c_bool(nesterov), stream.handle if stream else None)


def adagrad_update(param, grad, accumulation, lr, eps, l2reg, stream=None):
    assert isinstance(param, NDArray)
    assert isinstance(grad, (NDArray, IndexedSlices))
    assert isinstance(accumulation, NDArray)
    grad = add_l2_regularization(param, grad, l2reg, stream)
    if isinstance(grad, NDArray):
        _LIB.AdaGradOptimizerUpdate(param.handle, grad.handle, accumulation.handle, ctypes.c_float(
            lr), ctypes.c_float(eps), stream.handle if stream else None)
    else:
        grad.deduplicate(stream)
        assert isinstance(grad.dedup_ind, NDArray)
        assert isinstance(grad.dedup_val, NDArray)
        _LIB.AdaGradOptimizerSparseUpdate(param.handle, grad.dedup_ind.handle, grad.dedup_val.handle, accumulation.handle, ctypes.c_float(
            lr), ctypes.c_float(eps), stream.handle if stream else None)


def adam_update(param, grad, expavg, expavgsq, maxv, lr, beta1, beta2, beta1t, beta2t, eps, l2reg, stream=None):
    assert isinstance(param, NDArray)
    assert isinstance(grad, (NDArray, IndexedSlices))
    assert isinstance(expavg, NDArray)
    assert isinstance(expavgsq, NDArray)
    assert maxv is None or isinstance(maxv, NDArray)
    grad = add_l2_regularization(param, grad, l2reg, stream)
    if isinstance(grad, NDArray):
        _LIB.AdamOptimizerUpdate(param.handle, grad.handle, expavg.handle, expavgsq.handle, None if maxv is None else maxv.handle, ctypes.c_float(lr), ctypes.c_float(beta1), ctypes.c_float(beta2),
                                 ctypes.c_float(beta1t), ctypes.c_float(beta2t), ctypes.c_float(eps), stream.handle if stream else None)
    else:
        grad.deduplicate(stream)
        assert isinstance(grad.dedup_ind, NDArray)
        assert isinstance(grad.dedup_val, NDArray)
        _LIB.AdamOptimizerSparseUpdate(param.handle, grad.dedup_ind.handle, grad.dedup_val.handle, expavg.handle, expavgsq.handle, None if maxv is None else maxv.handle, ctypes.c_float(lr), ctypes.c_float(beta1), ctypes.c_float(beta2),
                                       ctypes.c_float(beta1t), ctypes.c_float(beta2t), ctypes.c_float(eps),  stream.handle if stream else None)


def adamw_update(param, grad, expavg, expavgsq, lr, beta1, beta2, beta1t, beta2t, eps, weight_decay, stream=None):
    assert isinstance(param, NDArray)
    assert isinstance(grad, (NDArray, IndexedSlices))
    assert isinstance(expavg, NDArray)
    assert isinstance(expavgsq, NDArray)
    if isinstance(grad, NDArray):
        _LIB.AdamWOptimizerUpdate(param.handle, grad.handle, expavg.handle, expavgsq.handle, ctypes.c_float(lr), ctypes.c_float(beta1), ctypes.c_float(beta2),
                                  ctypes.c_float(beta1t), ctypes.c_float(beta2t), ctypes.c_float(eps), ctypes.c_float(weight_decay), stream.handle if stream else None)
    else:
        grad.deduplicate(stream)
        assert isinstance(grad.indices, NDArray)
        assert isinstance(grad.values, NDArray)
        _LIB.AdamWOptimizerSparseUpdate(param.handle, grad.indices.handle, grad.values.handle, expavg.handle, expavgsq.handle, ctypes.c_float(lr), ctypes.c_float(beta1), ctypes.c_float(beta2),
                                        ctypes.c_float(beta1t), ctypes.c_float(beta2t), ctypes.c_float(eps), ctypes.c_float(weight_decay), stream.handle if stream else None)
        grad.free_deduplicate()


def lamb_update(param, grad, expavg, expavgsq, lr, beta1, beta2, beta1t, beta2t, eps, weight_decay, stream=None):
    assert isinstance(param, NDArray)
    assert isinstance(grad, (NDArray, IndexedSlices))
    assert isinstance(expavg, NDArray)
    assert isinstance(expavgsq, NDArray)
    if isinstance(grad, NDArray):
        _LIB.LambOptimizerUpdate(param.handle, grad.handle, expavg.handle, expavgsq.handle, ctypes.c_float(lr), ctypes.c_float(beta1), ctypes.c_float(beta2),
                                 ctypes.c_float(beta1t), ctypes.c_float(beta2t), ctypes.c_float(eps), ctypes.c_float(weight_decay), stream.handle if stream else None)
    else:
        grad.deduplicate(stream)
        assert isinstance(grad.indices, NDArray)
        assert isinstance(grad.values, NDArray)
        _LIB.LambOptimizerSparseUpdate(param.handle, grad.indices.handle, grad.values.handle, expavg.handle, expavgsq.handle, ctypes.c_float(lr), ctypes.c_float(beta1), ctypes.c_float(beta2),
                                       ctypes.c_float(beta1t), ctypes.c_float(beta2t), ctypes.c_float(eps), ctypes.c_float(weight_decay), stream.handle if stream else None)
        grad.free_deduplicate()
