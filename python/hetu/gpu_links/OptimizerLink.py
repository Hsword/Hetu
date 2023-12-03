from __future__ import absolute_import

import ctypes
from .._base import _LIB
from ..ndarray import NDArray, IndexedSlices


def add_l2_regularization(param, grad, l2reg, stream=None):
    if l2reg > 0:
        if isinstance(grad, IndexedSlices):
            grad = grad.to_dense(stream)
        _LIB.AddL2Regularization(param.handle, grad.handle, ctypes.c_float(
            l2reg), stream.handle if stream else None)
    return grad


def sgd_update(param, grad, lr, l2reg, stream=None):
    assert isinstance(param, NDArray)
    assert isinstance(grad, (NDArray, IndexedSlices))
    grad = add_l2_regularization(param, grad, l2reg, stream)
    if isinstance(grad, IndexedSlices):
        assert isinstance(grad.indices, NDArray)
        assert isinstance(grad.values, NDArray)
        _LIB.SGDOptimizerSparseUpdate(param.handle, grad.indices.handle, grad.values.handle, ctypes.c_float(
            lr), stream.handle if stream else None)
    else:
        _LIB.SGDOptimizerUpdate(param.handle, grad.handle, ctypes.c_float(
            lr), stream.handle if stream else None)


def sgd_update_indexedslices(indices, grads, params, output, lr, stream=None):
    assert isinstance(indices, NDArray)
    assert isinstance(grads, NDArray)
    assert isinstance(params, NDArray)
    assert isinstance(output, NDArray)
    _LIB.DLGpuSGDUpdateIndexedSlices(indices.handle, grads.handle, params.handle,
                                     output.handle, ctypes.c_float(lr), stream.handle if stream else None)


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
        assert isinstance(grad.indices, NDArray)
        assert isinstance(grad.values, NDArray)
        _LIB.AdaGradOptimizerSparseUpdate(param.handle, grad.indices.handle, grad.values.handle, accumulation.handle, ctypes.c_float(
            lr), ctypes.c_float(eps), stream.handle if stream else None)


def adagrad_update_indexedslices(indices, grads, params, output, lr, accum, epsilon, stream=None):
    assert isinstance(indices, NDArray)
    assert isinstance(grads, NDArray)
    assert isinstance(params, NDArray)
    assert isinstance(output, NDArray)
    assert isinstance(accum, NDArray)
    _LIB.DLGpuAdaGradUpdateIndexedSlices(
        indices.handle, grads.handle, params.handle,
        output.handle, ctypes.c_float(lr),
        accum.handle, ctypes.c_float(epsilon),
        stream.handle if stream else None)


def betats_update(betats, beta1, beta2, stream=None):
    assert isinstance(betats, NDArray)
    _LIB.BetatsUpdate(betats.handle, ctypes.c_float(beta1),
                      ctypes.c_float(beta2), stream.handle if stream else None)


def adam_update(param, grad, expavg, expavgsq, maxv, lr, beta1, beta2, betats, eps, l2reg, stream=None):
    assert isinstance(param, NDArray)
    assert isinstance(grad, (NDArray, IndexedSlices))
    assert isinstance(expavg, NDArray)
    assert isinstance(expavgsq, NDArray)
    assert isinstance(betats, NDArray)
    assert maxv is None or isinstance(maxv, NDArray)
    grad = add_l2_regularization(param, grad, l2reg, stream)
    if isinstance(grad, NDArray):
        _LIB.AdamOptimizerUpdate(param.handle, grad.handle, expavg.handle, expavgsq.handle, None if maxv is None else maxv.handle, ctypes.c_float(lr), ctypes.c_float(beta1), ctypes.c_float(beta2),
                                 betats.handle, ctypes.c_float(eps), stream.handle if stream else None)
    else:
        assert isinstance(grad.indices, NDArray)
        assert isinstance(grad.values, NDArray)
        _LIB.AdamOptimizerSparseUpdate(param.handle, grad.indices.handle, grad.values.handle, expavg.handle, expavgsq.handle, None if maxv is None else maxv.handle, ctypes.c_float(lr), ctypes.c_float(beta1), ctypes.c_float(beta2),
                                       betats.handle, ctypes.c_float(eps),  stream.handle if stream else None)


def adam_update_indexedslices(indices, grads, params, output, lr,
                              m, v, maxv, beta1, beta2, betats, epsilon, stream=None):
    assert isinstance(indices, NDArray)
    assert isinstance(grads, NDArray)
    assert isinstance(params, NDArray)
    assert isinstance(output, NDArray)
    assert isinstance(m, NDArray)
    assert isinstance(v, NDArray)
    assert maxv is None or isinstance(maxv, NDArray)
    assert isinstance(betats, NDArray)
    _LIB.DLGpuAdamUpdateIndexedSlices(
        indices.handle, grads.handle, params.handle,
        output.handle, ctypes.c_float(lr),
        m.handle, v.handle, maxv.handle if maxv else None,
        ctypes.c_float(beta1), ctypes.c_float(beta2),
        betats.handle, ctypes.c_float(epsilon),
        stream.handle if stream else None)


def adamw_update(param, grad, expavg, expavgsq, lr, beta1, beta2, betats, eps, weight_decay, stream=None):
    assert isinstance(param, NDArray)
    assert isinstance(grad, (NDArray, IndexedSlices))
    assert isinstance(expavg, NDArray)
    assert isinstance(expavgsq, NDArray)
    assert isinstance(betats, NDArray)
    if isinstance(grad, NDArray):
        _LIB.AdamWOptimizerUpdate(param.handle, grad.handle, expavg.handle, expavgsq.handle, ctypes.c_float(lr), ctypes.c_float(beta1), ctypes.c_float(beta2),
                                  betats.handle, ctypes.c_float(eps), ctypes.c_float(weight_decay), stream.handle if stream else None)
    else:
        assert isinstance(grad.indices, NDArray)
        assert isinstance(grad.values, NDArray)
        _LIB.AdamWOptimizerSparseUpdate(param.handle, grad.indices.handle, grad.values.handle, expavg.handle, expavgsq.handle, ctypes.c_float(lr), ctypes.c_float(beta1), ctypes.c_float(beta2),
                                        betats.handle, ctypes.c_float(eps), ctypes.c_float(weight_decay), stream.handle if stream else None)


def lamb_update(param, grad, expavg, expavgsq, lr, beta1, beta2, betats, eps, weight_decay, stream=None):
    assert isinstance(param, NDArray)
    assert isinstance(grad, (NDArray, IndexedSlices))
    assert isinstance(expavg, NDArray)
    assert isinstance(expavgsq, NDArray)
    assert isinstance(betats, NDArray)
    if isinstance(grad, NDArray):
        _LIB.LambOptimizerUpdate(param.handle, grad.handle, expavg.handle, expavgsq.handle, ctypes.c_float(lr), ctypes.c_float(beta1), ctypes.c_float(beta2),
                                 betats.handle, ctypes.c_float(eps), ctypes.c_float(weight_decay), stream.handle if stream else None)
    else:
        assert isinstance(grad.indices, NDArray)
        assert isinstance(grad.values, NDArray)
        _LIB.LambOptimizerSparseUpdate(param.handle, grad.indices.handle, grad.values.handle, expavg.handle, expavgsq.handle, ctypes.c_float(lr), ctypes.c_float(beta1), ctypes.c_float(beta2),
                                       betats.handle, ctypes.c_float(eps), ctypes.c_float(weight_decay), stream.handle if stream else None)
