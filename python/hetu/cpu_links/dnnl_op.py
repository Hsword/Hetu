from __future__ import absolute_import
import os
import ctypes
from .._base import _LIB
import numpy as np
from .. import ndarray as _nd


def matrix_multiply(matA, transposeA, matB, transposeB, matC):
    assert isinstance(matA, _nd.NDArray)
    assert isinstance(matB, _nd.NDArray)
    assert isinstance(matC, _nd.NDArray)
    _LIB.DnnlMatrixMultiply(matA.handle, transposeA,
                            matB.handle, transposeB, matC.handle)


def matrix_elementwise_multiply_by_const(mat, val, output):
    assert isinstance(mat, _nd.NDArray)
    assert isinstance(output, _nd.NDArray)
    _LIB.DnnlMatrixElementwiseMultiplyByConst(
        mat.handle, ctypes.c_float(val), output.handle)


def matrix_elementwise_multiply(matA, matB, output):
    assert isinstance(matA, _nd.NDArray)
    assert isinstance(matB, _nd.NDArray)
    assert isinstance(output, _nd.NDArray)
    _LIB.DnnlMatrixElementwiseMultiply(matA.handle, matB.handle, output.handle)


def matrix_elementwise_add_by_const(mat, val, output):
    assert isinstance(mat, _nd.NDArray)
    assert isinstance(output, _nd.NDArray)
    _LIB.DnnlMatrixElementwiseAddByConst(
        mat.handle, ctypes.c_float(val), output.handle)


def matrix_elementwise_add(matA, matB, output):
    assert isinstance(matA, _nd.NDArray)
    assert isinstance(matB, _nd.NDArray)
    assert isinstance(output, _nd.NDArray)
    _LIB.DnnlMatrixElementwiseAdd(matA.handle, matB.handle, output.handle)


def matrix_elementwise_divide_by_const(mat, val, output):
    assert isinstance(mat, _nd.NDArray)
    assert isinstance(output, _nd.NDArray)
    _LIB.DnnlMatrixElementwiseDivideByConst(
        mat.handle, ctypes.c_float(val), output.handle)


def matrix_elementwise_divide(matA, matB, output):
    assert isinstance(matA, _nd.NDArray)
    assert isinstance(matB, _nd.NDArray)
    assert isinstance(output, _nd.NDArray)
    _LIB.DnnlMatrixElementwiseDivide(matA.handle, matB.handle, output.handle)


def broadcast_to(in_arr, out_arr):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.cpu_BroadcastTo(in_arr.handle, out_arr.handle)


def reduce_sum_axis_zero(in_arr, out_arr):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.cpu_ReduceSumAxisZero(in_arr.handle, out_arr.handle)


def array_set(output, value):
    assert isinstance(output, _nd.NDArray)
    _LIB.cpu_ArraySet(output.handle, ctypes.c_float(value))


def reshape(in_arr, out_arr):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.cpu_Reshape(in_arr.handle, out_arr.handle)


def softmax(mat, output):
    assert isinstance(mat, _nd.NDArray)
    assert isinstance(output, _nd.NDArray)
    _LIB.DnnlSoftmax(mat.handle, output.handle)


def softmax_crossentropy(matA, matB, output):
    assert isinstance(matA, _nd.NDArray)
    assert isinstance(matB, _nd.NDArray)
    assert isinstance(output, _nd.NDArray)
    _LIB.DnnlSoftmaxCrossEntropy(matA.handle, matB.handle, output.handle)


def sqrt(in_arr, out_arr):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DnnlSqrt(in_arr.handle, out_arr.handle)


def rsqrt(in_arr, out_arr):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DnnlReciprocalSqrt(in_arr.handle, out_arr.handle)


def tanh(in_arr, out_arr):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DnnlTanh(in_arr.handle, out_arr.handle)


def opposite(in_arr, out_arr):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DnnlOpposite(in_arr.handle, out_arr.handle)


def sigmoid(in_arr, out_arr):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DnnlSigmoid(in_arr.handle, out_arr.handle)


def conv2d(input_x, input_f, output, padding=0, stride=1):
    assert isinstance(input_x, _nd.NDArray)
    assert isinstance(input_f, _nd.NDArray)
    assert isinstance(output, _nd.NDArray)
    _LIB.DnnlConv2d(input_x.handle, input_f.handle, output.handle,
                    ctypes.c_int(padding), ctypes.c_int(stride))


def conv2d_gradient_of_data(input_f, gradient_y, gradient_x, padding=0, stride=1):
    assert isinstance(gradient_y, _nd.NDArray)
    assert isinstance(input_f, _nd.NDArray)
    assert isinstance(gradient_x, _nd.NDArray)
    _LIB.DnnlConv2d_Gradient_of_Data(input_f.handle, gradient_y.handle, gradient_x.handle, ctypes.c_int(padding),
                                     ctypes.c_int(stride))


def conv2d_gradient_of_filter(input_x, gradient_y, gradient_f, padding=0, stride=1):
    assert isinstance(gradient_y, _nd.NDArray)
    assert isinstance(input_x, _nd.NDArray)
    assert isinstance(gradient_f, _nd.NDArray)
    _LIB.DnnlConv2d_Gradient_of_Filter(input_x.handle, gradient_y.handle, gradient_f.handle, ctypes.c_int(padding),
                                       ctypes.c_int(stride))


def avg_pool(input, H, W, output, padding=0, stride=1):
    assert isinstance(input, _nd.NDArray)
    assert isinstance(output, _nd.NDArray)
    _LIB.DnnlAvgPool(input.handle, ctypes.c_int(H), ctypes.c_int(W), output.handle, ctypes.c_int(padding),
                     ctypes.c_int(stride))


def avg_pool_gradient(gradient_Y, H, W, gradient_X, padding=0, stride=1):
    assert isinstance(gradient_Y, _nd.NDArray)
    assert isinstance(gradient_X, _nd.NDArray)
    _LIB.DnnlAvgPool_Gradient(gradient_Y.handle, ctypes.c_int(H), ctypes.c_int(W), gradient_X.handle,
                              ctypes.c_int(padding), ctypes.c_int(stride))


def max_pool(input, H, W, output, padding=0, stride=1):
    assert isinstance(input, _nd.NDArray)
    assert isinstance(output, _nd.NDArray)
    _LIB.DnnlMaxPool(input.handle, ctypes.c_int(H), ctypes.c_int(W), output.handle, ctypes.c_int(padding),
                     ctypes.c_int(stride))


def max_pool_gradient(input, input_grad, H, W, output, padding=0, stride=1):
    assert isinstance(input, _nd.NDArray)
    assert isinstance(output, _nd.NDArray)
    _LIB.DnnlMaxPool_Gradient(input.handle, input_grad.handle, ctypes.c_int(H), ctypes.c_int(W), output.handle,
                              ctypes.c_int(padding), ctypes.c_int(stride))


def relu(in_arr, out_arr):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DnnlRelu(in_arr.handle, out_arr.handle)


def relu_gradient(input, in_grad, output):
    assert isinstance(input, _nd.NDArray)
    assert isinstance(in_grad, _nd.NDArray)
    assert isinstance(output, _nd.NDArray)
    _LIB.DnnlRelu_Gradient(input.handle, in_grad.handle, output.handle)


def batch_norm(input, bn_scale, bn_bias, output, running_mean, running_var, save_mean, save_var, momentum=0.99, eps=0.01):
    assert isinstance(input, _nd.NDArray)
    assert isinstance(bn_scale, _nd.NDArray)
    assert isinstance(bn_bias, _nd.NDArray)
    assert isinstance(output, _nd.NDArray)
    assert isinstance(running_mean, _nd.NDArray)
    assert isinstance(running_var, _nd.NDArray)
    assert isinstance(save_mean, _nd.NDArray)
    assert isinstance(save_var, _nd.NDArray)
    _LIB.DnnlBatchNorm(input.handle, bn_scale.handle, bn_bias.handle, output.handle, running_mean.handle,
                       running_var.handle, save_mean.handle, save_var.handle, ctypes.c_float(momentum), ctypes.c_float(eps))


def batch_norm_gradient(gradient_Y, input_X, bn_scale, gradient_X, gradient_bn_scale, gradient_bn_bias, mean,
                        var, eps=0.01):
    assert isinstance(gradient_Y, _nd.NDArray)
    assert isinstance(input_X, _nd.NDArray)
    assert isinstance(gradient_X, _nd.NDArray)
    assert isinstance(gradient_bn_scale, _nd.NDArray)
    assert isinstance(gradient_bn_bias, _nd.NDArray)
    assert isinstance(bn_scale, _nd.NDArray)
    assert isinstance(mean, _nd.NDArray)
    assert isinstance(var, _nd.NDArray)
    _LIB.DnnlBatchNorm_Gradient(gradient_Y.handle, input_X.handle, bn_scale.handle,
                                gradient_X.handle, gradient_bn_scale.handle,
                                gradient_bn_bias.handle, mean.handle, var.handle, ctypes.c_float(eps))


def batch_norm_inference(input, bn_scale, bn_bias, output, mean, var, eps):
    assert isinstance(input, _nd.NDArray)
    assert isinstance(bn_scale, _nd.NDArray)
    assert isinstance(bn_bias, _nd.NDArray)
    assert isinstance(output, _nd.NDArray)
    assert isinstance(mean, _nd.NDArray)
    assert isinstance(var, _nd.NDArray)
    _LIB.DnnlBatchNorm_Inference(input.handle, bn_scale.handle, bn_bias.handle, output.handle,
                                 mean.handle, var.handle, ctypes.c_float(eps))


def concat(input_x, input_y, output, axis=0):
    assert isinstance(input_x, _nd.NDArray)
    assert isinstance(input_y, _nd.NDArray)
    assert isinstance(output, _nd.NDArray)
    _LIB.DnnlConcat(input_x.handle, input_y.handle,
                    output.handle, ctypes.c_int(axis))


def concat_gradient(output_gradient, input_gradient, axis=0, id=0):
    assert isinstance(output_gradient, _nd.NDArray)
    assert isinstance(input_gradient, _nd.NDArray)
    _LIB.cpu_Concat_Gradient(
        output_gradient.handle, input_gradient.handle, ctypes.c_int(axis), ctypes.c_int(id))


def dropout(in_arr, dropout, out_arr):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.cpu_Dropout(in_arr.handle, ctypes.c_float(dropout), out_arr.handle)


def dropout_gradient(in_gradient_y, dropout, out_gradient_x):
    assert isinstance(in_gradient_y, _nd.NDArray)
    assert isinstance(out_gradient_x, _nd.NDArray)
    _LIB.cpu_Dropout_Gradient(in_gradient_y.handle,
                              ctypes.c_float(dropout), out_gradient_x.handle)


def pad(in_arr, out_arr, paddings, mode='CONSTANT', constant_values=0):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    padding_arr = []
    for i in range(len(paddings)):
        for j in range(len(paddings[0])):
            padding_arr.append(paddings[i][j])
    pad_len = len(padding_arr)
    padding_c_arr = (ctypes.c_int * pad_len)(*padding_arr)
    f_type = 3
    if mode == 'CONSTANT':
        f_type = 0
    elif mode == 'REFLECT':
        f_type = 1
    elif mode == 'SYMMETRIC':
        f_type = 2
    assert (f_type <= 2)
    _LIB.cpu_Pad(in_arr.handle, out_arr.handle, padding_c_arr,
                 ctypes.c_int(pad_len), ctypes.c_int(f_type), ctypes.c_float(constant_values))


def pad_gradient(out_grad_arr, in_grad_arr, paddings, mode="CONSTANT"):
    assert isinstance(out_grad_arr, _nd.NDArray)
    assert isinstance(in_grad_arr, _nd.NDArray)
    padding_arr = []
    for i in range(len(paddings)):
        for j in range(len(paddings[0])):
            padding_arr.append(paddings[i][j])
    pad_len = len(padding_arr)
    padding_c_arr = (ctypes.c_int * pad_len)(*padding_arr)
    f_type = 3
    if mode == 'CONSTANT':
        f_type = 0
    elif mode == 'REFLECT':
        f_type = 1
    elif mode == 'SYMMETRIC':
        f_type = 2
    assert (f_type <= 2)
    _LIB.cpu_Pad_Gradient(out_grad_arr.handle,
                          in_grad_arr.handle, padding_c_arr, ctypes.c_int(pad_len), ctypes.c_int(f_type))


def transpose(in_arr, out_arr, perm):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    pointer_func = ctypes.c_int * len(perm)
    pointer = pointer_func(*list(perm))
    _LIB.cpu_Transpose(in_arr.handle, out_arr.handle, pointer)


def embedding_lookup(in_mat, ids, out_mat):
    assert isinstance(in_mat, _nd.NDArray)
    assert isinstance(ids, _nd.NDArray)
    assert isinstance(out_mat, _nd.NDArray)
    _LIB.cpu_EmbeddingLookup(in_mat.handle, ids.handle, out_mat.handle)


def add_l2_regularization(param, grad, l2reg):
    assert isinstance(param, _nd.NDArray)
    assert isinstance(grad, _nd.NDArray)
    _LIB.cpu_AddL2Regularization(
        param.handle, grad.handle, ctypes.c_float(l2reg))


def sgd_update(param, grad, lr):
    assert isinstance(param, _nd.NDArray)
    assert isinstance(grad, _nd.NDArray)
    _LIB.cpu_SGDOptimizerUpdate(param.handle, grad.handle, ctypes.c_float(lr))


def sgd_update_sparse(param, grad_ind, grad_value, lr):
    assert isinstance(param, _nd.NDArray)
    assert isinstance(grad_ind, _nd.NDArray)
    assert isinstance(grad_value, _nd.NDArray)
    _LIB.cpu_SGDOptimizerSparseUpdate(
        param.handle, grad_ind.handle, grad_value.handle, ctypes.c_float(lr))


def momentum_update(param, grad, velocity, lr, momentum, nesterov):
    assert isinstance(param, _nd.NDArray)
    assert isinstance(grad, _nd.NDArray)
    assert isinstance(velocity, _nd.NDArray)
    _LIB.cpu_MomentumOptimizerUpdate(param.handle, grad.handle, velocity.handle,
                                     ctypes.c_float(lr), ctypes.c_float(momentum), ctypes.c_bool(nesterov))


def adagrad_update(param, grad, accumulation, lr, eps):
    assert isinstance(param, _nd.NDArray)
    assert isinstance(grad, _nd.NDArray)
    assert isinstance(accumulation, _nd.NDArray)
    _LIB.cpu_AdaGradOptimizerUpdate(param.handle, grad.handle, accumulation.handle,
                                    ctypes.c_float(lr), ctypes.c_float(eps))


def adam_update(param, grad, expavg, expavgsq, lr, beta1, beta2, beta1t, beta2t, eps):
    assert isinstance(param, _nd.NDArray)
    assert isinstance(grad, _nd.NDArray)
    assert isinstance(expavg, _nd.NDArray)
    assert isinstance(expavgsq, _nd.NDArray)
    _LIB.cpu_AdamOptimizerUpdate(param.handle, grad.handle, expavg.handle,
                                 expavgsq.handle, ctypes.c_float(lr),
                                 ctypes.c_float(beta1), ctypes.c_float(
                                     beta2), ctypes.c_float(beta1t),
                                 ctypes.c_float(beta2t), ctypes.c_float(eps))


def normal_init(param, mean, stddev, seed):
    assert isinstance(param, _nd.NDArray)
    _LIB.cpu_NormalInit(param.handle, ctypes.c_float(
        mean), ctypes.c_float(stddev), ctypes.c_ulonglong(seed))


def uniform_init(param, lb, ub, seed):
    assert isinstance(param, _nd.NDArray)
    _LIB.cpu_UniformInit(param.handle, ctypes.c_float(
        lb), ctypes.c_float(ub), ctypes.c_ulonglong(seed))


def truncated_normal_init(param, mean, stddev, seed):
    assert isinstance(param, _nd.NDArray)
    _LIB.cpu_TruncatedNormalInit(param.handle, ctypes.c_float(
        mean), ctypes.c_float(stddev), ctypes.c_ulonglong(seed))


def gelu(in_arr, out_arr):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    _LIB.DnnlGelu(in_arr.handle, out_arr.handle)


def gelu_gradient(input, in_grad, output):
    assert isinstance(input, _nd.NDArray)
    assert isinstance(in_grad, _nd.NDArray)
    assert isinstance(output, _nd.NDArray)
    _LIB.DnnlGelu_Gradient(input.handle, in_grad.handle, output.handle)
