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
                            float lr, float beta1, float beta2, float *betats,
                            float eps, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    m[ind] = beta1 * m[ind] + (1 - beta1) * grad[ind];
    v[ind] = beta2 * v[ind] + (1 - beta2) * grad[ind] * grad[ind];
    float m_local = m[ind] / (1 - betats[0]);
    float v_local = v[ind] / (1 - betats[1]);
    param[ind] = param[ind] - lr * m_local / (sqrtf(v_local) + eps);
}

__global__ void amsgrad_update(float *param, const float *grad, float *m,
                               float *v, float *maxv, float lr, float beta1,
                               float beta2, float *betats, float eps,
                               size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    m[ind] = beta1 * m[ind] + (1 - beta1) * grad[ind];
    v[ind] = beta2 * v[ind] + (1 - beta2) * grad[ind] * grad[ind];
    float m_local = m[ind] / (1 - betats[0]);
    float v_local = v[ind] / (1 - betats[1]);
    float cur_maxv = fmaxf(v_local, maxv[ind]);
    maxv[ind] = cur_maxv;
    param[ind] = param[ind] - lr * m_local / (sqrtf(cur_maxv) + eps);
}

__global__ void update_betats(float *betats, float beta1, float beta2) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind == 0) {
        betats[0] *= beta1;
        betats[1] *= beta2;
    }
}

int BetatsUpdate(DLArrayHandle betats, float beta1, float beta2,
                 DLStreamHandle stream_handle = NULL) {
    float *betats_data = (float *)betats->data;
    if (stream_handle) {
        update_betats<<<1, 1, 0, *(cudaStream_t *)stream_handle->handle>>>(
            betats_data, beta1, beta2);
    } else {
        update_betats<<<1, 1>>>(betats_data, beta1, beta2);
    }
    return 0;
}

int AdamOptimizerUpdate(DLArrayHandle param, const DLArrayHandle grad,
                        DLArrayHandle expavg, DLArrayHandle expavgsq,
                        DLArrayHandle maxv, float lr, float beta1, float beta2,
                        DLArrayHandle betats, float eps,
                        DLStreamHandle stream_handle = NULL) {
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
    float *betats_data = (float *)betats->data;
    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }
    if (maxv != NULL) {
        float *maxv_data = (float *)maxv->data;
        if (stream_handle) {
            amsgrad_update<<<blocks, threads, 0,
                             *(cudaStream_t *)stream_handle->handle>>>(
                param_data, grad_data, m_data, v_data, maxv_data, lr, beta1,
                beta2, betats_data, eps, size);
        } else {
            amsgrad_update<<<blocks, threads>>>(param_data, grad_data, m_data,
                                                v_data, maxv_data, lr, beta1,
                                                beta2, betats_data, eps, size);
        }

    } else {
        if (stream_handle) {
            adam_update<<<blocks, threads, 0,
                          *(cudaStream_t *)stream_handle->handle>>>(
                param_data, grad_data, m_data, v_data, lr, beta1, beta2,
                betats_data, eps, size);
        } else {
            adam_update<<<blocks, threads>>>(param_data, grad_data, m_data,
                                             v_data, lr, beta1, beta2,
                                             betats_data, eps, size);
        }
    }
    return 0;
}

__global__ void adamw_update(float *param, const float *grad, float *m,
                             float *v, float lr, float beta1, float beta2,
                             float *betats, float eps, float weight_decay,
                             size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    m[ind] = beta1 * m[ind] + (1 - beta1) * grad[ind];
    v[ind] = beta2 * v[ind] + (1 - beta2) * grad[ind] * grad[ind];
    float m_local = m[ind] / (1 - betats[0]);
    float v_local = v[ind] / (1 - betats[1]);
    float update = m_local / (sqrtf(v_local) + eps);
    param[ind] = param[ind] - lr * (update + weight_decay * param[ind]);
}

int AdamWOptimizerUpdate(DLArrayHandle param, const DLArrayHandle grad,
                         DLArrayHandle expavg, DLArrayHandle expavgsq, float lr,
                         float beta1, float beta2, DLArrayHandle betats,
                         float eps, float weight_decay,
                         DLStreamHandle stream_handle = NULL) {
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
    float *betats_data = (float *)betats->data;
    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }
    if (stream_handle)
        adamw_update<<<blocks, threads, 0,
                       *(cudaStream_t *)stream_handle->handle>>>(
            param_data, grad_data, m_data, v_data, lr, beta1, beta2,
            betats_data, eps, weight_decay, size);
    else
        adamw_update<<<blocks, threads>>>(param_data, grad_data, m_data, v_data,
                                          lr, beta1, beta2, betats_data, eps,
                                          weight_decay, size);
    return 0;
}

__global__ void calc_lamb_update(float *update, const float *grad, float *m,
                                 float *v, float beta1, float beta2,
                                 float *betats, float eps, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    m[ind] = beta1 * m[ind] + (1 - beta1) * grad[ind];
    v[ind] = beta2 * v[ind] + (1 - beta2) * grad[ind] * grad[ind];
    float m_local = m[ind] / (1 - betats[0]);
    float v_local = v[ind] / (1 - betats[1]);
    update[ind] = m_local / (sqrtf(v_local) + eps);
}

__global__ void lamb_update_step(float *param, const float *update, float lr,
                                 float weight_decay, float *norm2_param,
                                 float *norm2_update, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    param[ind] = param[ind]
                 - lr * (norm2_param[0] / norm2_update[0])
                       * (update[ind] + weight_decay * param[ind]);
}

int LambOptimizerUpdate(DLArrayHandle param, const DLArrayHandle grad,
                        DLArrayHandle expavg, DLArrayHandle expavgsq, float lr,
                        float beta1, float beta2, DLArrayHandle betats,
                        float eps, float weight_decay,
                        DLStreamHandle stream_handle = NULL) {
    int dev_id = (param->ctx).device_id;
    cudaSetDevice(dev_id);
    cudnn_init(dev_id, stream_handle);

    // Prepare cudnn reduce tensor for Norm2 calculation of param and update
    float one = 1.0f;
    float zero = 0.0f;

    cudnnReduceTensorDescriptor_t rtd;
    CUDNN_CALL(cudnnCreateReduceTensorDescriptor(&rtd));
    CUDNN_CALL(cudnnSetReduceTensorDescriptor(
        rtd, CUDNN_REDUCE_TENSOR_NORM2, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN,
        CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES));

    cudnnTensorDescriptor_t adesc;
    cudnnTensorDescriptor_t cdesc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&adesc));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&cdesc));

    int ndim = param->ndim;
    if (ndim < 4)
        ndim = 4;
    size_t cpu_mem = ndim * sizeof(int);
    int *dimA = (int *)malloc(cpu_mem);
    int *strideA = (int *)malloc(cpu_mem);
    int *dimC = (int *)malloc(cpu_mem);
    int *strideC = (int *)malloc(cpu_mem);

    int temp_strideA = 1;
    int temp_strideC = 1;

    for (int i = ndim - 1; i >= 0; --i) {
        dimA[i] = i < param->ndim ? (int)param->shape[i] : 1;
        dimC[i] = 1;
        strideA[i] = temp_strideA;
        strideC[i] = temp_strideC;
        temp_strideA *= dimA[i];
        temp_strideC *= dimC[i];
    }
    size_t size = temp_strideA * sizeof(float);

    CUDNN_CALL(cudnnSetTensorNdDescriptor(adesc, CUDNN_DATA_FLOAT, ndim, dimA,
                                          strideA));
    CUDNN_CALL(cudnnSetTensorNdDescriptor(cdesc, CUDNN_DATA_FLOAT, ndim, dimC,
                                          strideC));

    dim3 blocks;
    dim3 threads;
    float *param_data = (float *)param->data;
    const float *grad_data = (const float *)grad->data;
    float *m_data = (float *)expavg->data;
    float *v_data = (float *)expavgsq->data;
    float *betats_data = (float *)betats->data;
    if (temp_strideA <= 1024) {
        threads.x = temp_strideA;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (temp_strideA + 1023) / 1024;
    }

    if (is_chunk_init(dev_id) == false) {
        chunk_init(dev_id);
    }
    void *norm2_param = find_chunk(1 * sizeof(float), dev_id);
    void *norm2_update = find_chunk(1 * sizeof(float), dev_id);
    void *workspace = find_chunk(size, dev_id);
    void *update = find_chunk(size, dev_id);

    // Calculate Norm2 of param
    CUDNN_CALL(cudnnReduceTensor(cudnn_map[dev_id], rtd, NULL, 0, workspace,
                                 size, &one, adesc, (const void *)param_data,
                                 &zero, cdesc, norm2_param));

    // Calculate update
    if (stream_handle)
        calc_lamb_update<<<blocks, threads, 0,
                           *(cudaStream_t *)stream_handle->handle>>>(
            (float *)update, grad_data, m_data, v_data, beta1, beta2,
            betats_data, eps, temp_strideA);
    else
        calc_lamb_update<<<blocks, threads>>>((float *)update, grad_data,
                                              m_data, v_data, beta1, beta2,
                                              betats_data, eps, temp_strideA);

    // Calculate Norm2 of update
    CUDNN_CALL(cudnnReduceTensor(cudnn_map[dev_id], rtd, NULL, 0, workspace,
                                 size, &one, adesc, (const void *)update, &zero,
                                 cdesc, norm2_update));

    // Update step
    if (stream_handle)
        lamb_update_step<<<blocks, threads, 0,
                           *(cudaStream_t *)stream_handle->handle>>>(
            (float *)param_data, (const float *)update, lr, weight_decay,
            (float *)norm2_param, (float *)norm2_update, size);
    else
        lamb_update_step<<<blocks, threads>>>(
            (float *)param_data, (const float *)update, lr, weight_decay,
            (float *)norm2_param, (float *)norm2_update, size);

    del_chunk(norm2_param, dev_id);
    del_chunk(norm2_update, dev_id);
    del_chunk(workspace, dev_id);
    del_chunk(update, dev_id);
    CUDNN_CALL(cudnnDestroyTensorDescriptor(adesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(cdesc));
    CUDNN_CALL(cudnnDestroyReduceTensorDescriptor(rtd));
    free(dimA);
    free(dimC);
    free(strideA);
    free(strideC);
    return 0;
}