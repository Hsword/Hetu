#include "gpu_runtime.h"

__global__ void add_l2_regularization(const float *param, float *grad,
                                      float l2reg, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    grad[ind] = grad[ind] + l2reg * param[ind];
}

int AddL2Regularization(const DLArrayHandle param, DLArrayHandle grad,
                        float l2reg, DLStreamHandle stream_handle = NULL) {
    size_t size = 1;
    for (index_t i = 0; i < param->ndim; ++i) {
        size *= param->shape[i];
    }
    dim3 blocks;
    dim3 threads;
    const float *param_data = (const float *)param->data;
    float *grad_data = (float *)grad->data;
    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }
    if (stream_handle) {
        add_l2_regularization<<<blocks, threads, 0,
                                *(cudaStream_t *)stream_handle->handle>>>(
            param_data, grad_data, l2reg, size);
    } else {
        add_l2_regularization<<<blocks, threads>>>(param_data, grad_data, l2reg,
                                                   size);
    }
    return 0;
}

__global__ void sgd_update(float *param, const float *grad, float lr,
                           size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    param[ind] = param[ind] - lr * grad[ind];
}

int SGDOptimizerUpdate(DLArrayHandle param, const DLArrayHandle grad, float lr,
                       DLStreamHandle stream_handle = NULL) {
    size_t size = 1;
    for (index_t i = 0; i < param->ndim; ++i) {
        size *= param->shape[i];
    }
    dim3 blocks;
    dim3 threads;
    float *param_data = (float *)param->data;
    const float *grad_data = (const float *)grad->data;
    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }
    if (stream_handle)
        sgd_update<<<blocks, threads, 0,
                     *(cudaStream_t *)stream_handle->handle>>>(
            param_data, grad_data, lr, size);
    else
        sgd_update<<<blocks, threads>>>(param_data, grad_data, lr, size);
    return 0;
}

__global__ void nesterov_momentum_update(float *param, const float *grad,
                                         float *velocity, float lr,
                                         float momentum, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    float temp = lr * grad[ind];
    velocity[ind] = momentum * (velocity[ind] - temp);
    param[ind] = param[ind] + velocity[ind] - temp;
}

__global__ void momentum_update(float *param, const float *grad,
                                float *velocity, float lr, float momentum,
                                size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    velocity[ind] = momentum * velocity[ind] - lr * grad[ind];
    param[ind] = param[ind] + velocity[ind];
}

int MomentumOptimizerUpdate(DLArrayHandle param, const DLArrayHandle grad,
                            DLArrayHandle velocity, float lr, float momentum,
                            bool nesterov,
                            DLStreamHandle stream_handle = NULL) {
    size_t size = 1;
    for (index_t i = 0; i < param->ndim; ++i) {
        size *= param->shape[i];
    }
    dim3 blocks;
    dim3 threads;
    float *param_data = (float *)param->data;
    const float *grad_data = (const float *)grad->data;
    float *velocity_data = (float *)velocity->data;
    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }
    if (nesterov) {
        if (stream_handle)
            nesterov_momentum_update<<<
                blocks, threads, 0, *(cudaStream_t *)stream_handle->handle>>>(
                param_data, grad_data, velocity_data, lr, momentum, size);
        else
            nesterov_momentum_update<<<blocks, threads>>>(
                param_data, grad_data, velocity_data, lr, momentum, size);
    } else {
        if (stream_handle)
            momentum_update<<<blocks, threads, 0,
                              *(cudaStream_t *)stream_handle->handle>>>(
                param_data, grad_data, velocity_data, lr, momentum, size);
        else
            momentum_update<<<blocks, threads>>>(
                param_data, grad_data, velocity_data, lr, momentum, size);
    }
    return 0;
}

__global__ void adagrad_update(float *param, const float *grad, float *acc,
                               float lr, float eps, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    acc[ind] = acc[ind] + grad[ind] * grad[ind];
    param[ind] = param[ind] - lr * grad[ind] / (sqrtf(acc[ind]) + eps);
}

int AdaGradOptimizerUpdate(DLArrayHandle param, const DLArrayHandle grad,
                           DLArrayHandle acc, float lr, float eps,
                           DLStreamHandle stream_handle = NULL) {
    size_t size = 1;
    for (index_t i = 0; i < param->ndim; ++i) {
        size *= param->shape[i];
    }
    dim3 blocks;
    dim3 threads;
    float *param_data = (float *)param->data;
    const float *grad_data = (const float *)grad->data;
    float *acc_data = (float *)acc->data;
    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }
    if (stream_handle)
        adagrad_update<<<blocks, threads, 0,
                         *(cudaStream_t *)stream_handle->handle>>>(
            param_data, grad_data, acc_data, lr, eps, size);
    else
        adagrad_update<<<blocks, threads>>>(param_data, grad_data, acc_data, lr,
                                            eps, size);
    return 0;
}

__global__ void adam_update(float *param, const float *grad, float *m, float *v,
                            float lr, float beta1, float beta2, float beta1t,
                            float beta2t, float eps, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    m[ind] = beta1 * m[ind] + (1 - beta1) * grad[ind];
    v[ind] = beta2 * v[ind] + (1 - beta2) * grad[ind] * grad[ind];
    float m_local = m[ind] / (1 - beta1t);
    float v_local = v[ind] / (1 - beta2t);
    param[ind] = param[ind] - lr * m_local / (sqrtf(v_local) + eps);
}

int AdamOptimizerUpdate(DLArrayHandle param, const DLArrayHandle grad,
                        DLArrayHandle expavg, DLArrayHandle expavgsq, float lr,
                        float beta1, float beta2, float beta1t, float beta2t,
                        float eps, DLStreamHandle stream_handle = NULL) {
    size_t size = 1;
    for (index_t i = 0; i < param->ndim; ++i) {
        size *= param->shape[i];
    }
    dim3 blocks;
    dim3 threads;
    float *param_data = (float *)param->data;
    const float *grad_data = (const float *)grad->data;
    float *m_data = (float *)expavg->data;
    float *v_data = (float *)expavgsq->data;
    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }
    if (stream_handle)
        adam_update<<<blocks, threads, 0,
                      *(cudaStream_t *)stream_handle->handle>>>(
            param_data, grad_data, m_data, v_data, lr, beta1, beta2, beta1t,
            beta2t, eps, size);
    else
        adam_update<<<blocks, threads>>>(param_data, grad_data, m_data, v_data,
                                         lr, beta1, beta2, beta1t, beta2t, eps,
                                         size);
    return 0;
}
