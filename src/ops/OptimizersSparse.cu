#include "gpu_runtime.h"

__global__ void add_l2_regularization_sparse(const float *param,
                                             float *grad_data,
                                             const int *indices_data,
                                             float l2reg, size_t size,
                                             size_t length) {
    size_t thread_ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_ind >= size)
        return;
    size_t ind = thread_ind / length;
    size_t offset = thread_ind % length;
    int grad_ind = indices_data[ind];
    if (grad_ind < 0)
        return;
    const float cur_grad = grad_data[thread_ind];
    size_t total_offset = length * grad_ind + offset;
    const float *param_ptr = param + total_offset;
    grad_data[thread_ind] = cur_grad + l2reg * (*param_ptr);
}

int AddL2RegularizationSparse(const DLArrayHandle param,
                              const DLArrayHandle grad_indices,
                              DLArrayHandle grad_values, float l2reg,
                              DLStreamHandle stream_handle = NULL) {
    size_t size = 1;
    size_t length = param->shape[1];
    for (int i = 0; i < grad_values->ndim; i++) {
        size *= grad_values->shape[i];
    }

    dim3 blocks;
    dim3 threads;
    float *grad_data = (float *)grad_values->data;
    const float *param_data = (const float *)param->data;
    const int *indices_data = (const int *)grad_indices->data;

    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }

    if (stream_handle)
        add_l2_regularization_sparse<<<
            blocks, threads, 0, *(cudaStream_t *)stream_handle->handle>>>(
            param_data, grad_data, indices_data, l2reg, size, length);
    else
        add_l2_regularization_sparse<<<blocks, threads>>>(
            param_data, grad_data, indices_data, l2reg, size, length);
    return 0;
}

__global__ void sgd_sparse_update(const float *grad_data,
                                  const int *indices_data, float *param_data,
                                  size_t size, size_t length, float lr) {
    size_t thread_ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_ind >= size)
        return;
    size_t index = thread_ind / length;
    size_t offset = thread_ind % length;
    int id = indices_data[index];
    if (id < 0)
        return;
    const float cur_grad = grad_data[thread_ind];
    param_data[length * id + offset] -= lr * cur_grad;
}

int SGDOptimizerSparseUpdate(DLArrayHandle param,
                             const DLArrayHandle grad_indices,
                             const DLArrayHandle grad_values, float lr,
                             DLStreamHandle stream_handle = NULL) {
    size_t size = ArrSize(grad_values);
    size_t length = param->shape[1];

    const float *grad_data = (const float *)grad_values->data;
    float *param_data = (float *)param->data;
    const int *indices_data = (const int *)grad_indices->data;

    dim3 blocks;
    dim3 threads;
    ThreadBlock1D(threads, blocks, size);

    if (stream_handle)
        sgd_sparse_update<<<blocks, threads, 0,
                            *(cudaStream_t *)stream_handle->handle>>>(
            grad_data, indices_data, param_data, size, length, lr);
    else
        sgd_sparse_update<<<blocks, threads>>>(grad_data, indices_data,
                                               param_data, size, length, lr);
    return 0;
}

__global__ void nesterov_sparse_first_phase(float *param_data,
                                            float *veloc_data,
                                            const float *grad_data,
                                            const int *indices_data, float lr,
                                            size_t size, size_t length) {
    size_t thread_ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_ind >= size)
        return;
    size_t ind = thread_ind / length;
    size_t offset = thread_ind % length;
    int grad_ind = indices_data[ind];
    if (grad_ind < 0)
        return;
    const float cur_grad = grad_data[thread_ind];
    size_t total_offset = length * grad_ind + offset;
    float *veloc_ptr = veloc_data + total_offset;
    float *param_ptr = param_data + total_offset;
    float temp = -lr * cur_grad;
    atomicAdd(veloc_ptr, temp);
    atomicAdd(param_ptr, temp);
}

__global__ void nesterov_sparse_second_phase(float *param_data,
                                             float *veloc_data, float momentum,
                                             size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    float temp_veloc = momentum * veloc_data[ind];
    veloc_data[ind] = temp_veloc;
    param_data[ind] = param_data[ind] + temp_veloc;
}

__global__ void momentum_sparse_first_phase(float *veloc_data,
                                            const float *grad_data,
                                            const int *indices_data, float lr,
                                            size_t size, size_t length) {
    size_t thread_ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_ind >= size)
        return;
    size_t ind = thread_ind / length;
    size_t offset = thread_ind % length;
    int grad_ind = indices_data[ind];
    if (grad_ind < 0)
        return;
    const float cur_grad = grad_data[thread_ind];
    float *veloc_ptr = veloc_data + length * grad_ind + offset;
    atomicAdd(veloc_ptr, -lr * cur_grad);
}

__global__ void momentum_sparse_second_phase(float *param_data,
                                             float *veloc_data, float momentum,
                                             size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    param_data[ind] = param_data[ind] + veloc_data[ind];
    veloc_data[ind] = momentum * veloc_data[ind];
}

// the following method is not correct; use to_dense and dense update
int MomentumOptimizerSparseUpdate(DLArrayHandle param,
                                  const DLArrayHandle grad_indices,
                                  const DLArrayHandle grad_values,
                                  DLArrayHandle velocity, float lr,
                                  float momentum, bool nesterov,
                                  DLStreamHandle stream_handle = NULL) {
    size_t size = 1;
    size_t total_size = 1;
    size_t length = param->shape[1];
    for (int i = 0; i < grad_values->ndim; ++i) {
        size *= grad_values->shape[i];
    }
    for (int i = 0; i < param->ndim; ++i) {
        total_size *= param->shape[i];
    }

    float *param_data = (float *)param->data;
    const float *grad_data = (const float *)grad_values->data;
    const int *indices_data = (const int *)grad_indices->data;
    float *velocity_data = (float *)velocity->data;
    dim3 blocks;
    dim3 threads;
    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }

    dim3 dense_blocks;
    dim3 dense_threads;
    if (total_size <= 1024) {
        dense_threads.x = total_size;
        dense_blocks.x = 1;
    } else {
        dense_threads.x = 1024;
        dense_blocks.x = (total_size + 1023) / 1024;
    }

    if (nesterov) {
        if (stream_handle) {
            nesterov_sparse_first_phase<<<
                blocks, threads, 0, *(cudaStream_t *)stream_handle->handle>>>(
                param_data, velocity_data, grad_data, indices_data, lr, size,
                length);
            nesterov_sparse_second_phase<<<dense_blocks, dense_threads, 0,
                                           *(cudaStream_t *)
                                                stream_handle->handle>>>(
                param_data, velocity_data, momentum, total_size);
        } else {
            nesterov_sparse_first_phase<<<blocks, threads>>>(
                param_data, velocity_data, grad_data, indices_data, lr, size,
                length);
            nesterov_sparse_second_phase<<<dense_blocks, dense_threads>>>(
                param_data, velocity_data, momentum, total_size);
        }
    } else {
        if (stream_handle) {
            momentum_sparse_first_phase<<<
                blocks, threads, 0, *(cudaStream_t *)stream_handle->handle>>>(
                velocity_data, grad_data, indices_data, lr, size, length);
            momentum_sparse_second_phase<<<dense_blocks, dense_threads, 0,
                                           *(cudaStream_t *)
                                                stream_handle->handle>>>(
                param_data, velocity_data, momentum, total_size);
        } else {
            momentum_sparse_first_phase<<<blocks, threads>>>(
                velocity_data, grad_data, indices_data, lr, size, length);
            momentum_sparse_second_phase<<<dense_blocks, dense_threads>>>(
                param_data, velocity_data, momentum, total_size);
        }
    }
    return 0;
}

__global__ void indexedslices2dense_kernel(const float *values_data,
                                           const int *indices_data,
                                           float *new_values_data, size_t size,
                                           size_t length) {
    size_t thread_ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_ind >= size)
        return;
    size_t ind = thread_ind / length;
    size_t offset = thread_ind % length;
    int to_ind = indices_data[ind];
    if (to_ind < 0)
        return;
    const float cur_value = values_data[thread_ind];
    float *new_value_ptr = new_values_data + length * to_ind + offset;
    *new_value_ptr = cur_value;
}

int IndexedSlices2Dense(const DLArrayHandle values, const DLArrayHandle indices,
                        DLArrayHandle new_values,
                        DLStreamHandle stream_handle) {
    size_t size = 1;
    size_t length = new_values->shape[new_values->ndim - 1];
    for (int i = 0; i < values->ndim; ++i) {
        size *= values->shape[i];
    }
    const float *values_data = (const float *)values->data;
    const int *indices_data = (const int *)indices->data;
    float *new_values_data = (float *)new_values->data;

    dim3 blocks;
    dim3 threads;
    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }

    if (stream_handle)
        indexedslices2dense_kernel<<<blocks, threads, 0,
                                     *(cudaStream_t *)stream_handle->handle>>>(
            values_data, indices_data, new_values_data, size, length);
    else
        indexedslices2dense_kernel<<<blocks, threads>>>(
            values_data, indices_data, new_values_data, size, length);

    return 0;
}

__global__ void deduplicate_kernel(const float *origin_data,
                                   const float *inverse_data,
                                   float *compressed_data, size_t size,
                                   size_t length) {
    size_t thread_ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_ind >= size)
        return;
    size_t ind = thread_ind / length;
    size_t offset = thread_ind % length;
    int com_ind = inverse_data[ind];
    const float cur_origin = origin_data[thread_ind];
    float *compressed_ptr = compressed_data + length * com_ind + offset;
    atomicAdd(compressed_ptr, cur_origin);
}

int DeduplicateIndexedSlices(const DLArrayHandle origin,
                             const DLArrayHandle inverse,
                             DLArrayHandle compressed,
                             DLStreamHandle stream_handle = NULL) {
    size_t size = 1;
    size_t length = compressed->shape[compressed->ndim - 1];
    for (int i = 0; i < origin->ndim; ++i) {
        size *= origin->shape[i];
    }
    const float *origin_data = (const float *)origin->data;
    const float *inverse_data = (const float *)inverse->data;
    float *compressed_data = (float *)compressed->data;

    dim3 blocks;
    dim3 threads;
    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }

    if (stream_handle)
        deduplicate_kernel<<<blocks, threads, 0,
                             *(cudaStream_t *)stream_handle->handle>>>(
            origin_data, inverse_data, compressed_data, size, length);
    else
        deduplicate_kernel<<<blocks, threads>>>(origin_data, inverse_data,
                                                compressed_data, size, length);

    return 0;
}

__global__ void adagrad_sparse_update(float *param_data, const float *grad_data,
                                      const int *indices_data, float *acc_data,
                                      float lr, float eps, size_t size,
                                      size_t length) {
    size_t thread_ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_ind >= size)
        return;
    size_t ind = thread_ind / length;
    size_t offset = thread_ind % length;

    int grad_ind = indices_data[ind];
    if (grad_ind < 0)
        return;
    const float cur_grad = grad_data[thread_ind];
    size_t total_offset = length * grad_ind + offset;
    float *acc_ptr = acc_data + total_offset;
    float *param_ptr = param_data + total_offset;

    float cur_acc = *acc_ptr + cur_grad * cur_grad;
    *acc_ptr = cur_acc;
    *param_ptr -= lr * cur_grad / (sqrtf(cur_acc) + eps);
}

int AdaGradOptimizerSparseUpdate(DLArrayHandle param,
                                 const DLArrayHandle grad_indices,
                                 const DLArrayHandle grad_values,
                                 DLArrayHandle acc, float lr, float eps,
                                 DLStreamHandle stream_handle = NULL) {
    size_t size = ArrSize(grad_values);
    size_t length = param->shape[1];

    float *param_data = (float *)param->data;
    const float *grad_data = (const float *)grad_values->data;
    const int *indices_data = (const int *)grad_indices->data;
    float *acc_data = (float *)acc->data;

    dim3 blocks;
    dim3 threads;
    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }

    if (stream_handle) {
        adagrad_sparse_update<<<blocks, threads, 0,
                                *(cudaStream_t *)stream_handle->handle>>>(
            param_data, grad_data, indices_data, acc_data, lr, eps, size,
            length);
    } else {
        adagrad_sparse_update<<<blocks, threads>>>(param_data, grad_data,
                                                   indices_data, acc_data, lr,
                                                   eps, size, length);
    }
    return 0;
}

__global__ void adam_sparse_update(float *param, const float *grad_data,
                                   const int *indices_data, float *m, float *v,
                                   float lr, float beta1, float beta2,
                                   float *betats, float eps, size_t size,
                                   size_t length) {
    size_t thread_ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_ind >= size)
        return;
    size_t ind = thread_ind / length;
    size_t offset = thread_ind % length;

    int grad_ind = indices_data[ind];
    if (grad_ind < 0)
        return;
    const float cur_grad = grad_data[thread_ind];
    size_t total_offset = length * grad_ind + offset;
    float *m_ptr = m + total_offset;
    float *v_ptr = v + total_offset;
    float *param_ptr = param + total_offset;

    float cur_m = beta1 * (*m_ptr) + (1 - beta1) * cur_grad;
    float cur_v = beta2 * (*v_ptr) + (1 - beta2) * cur_grad * cur_grad;
    *m_ptr = cur_m;
    *v_ptr = cur_v;
    cur_m /= (1 - betats[0]);
    cur_v /= (1 - betats[1]);
    *(param_ptr) -= lr * cur_m / (sqrtf(cur_v) + eps);
}

__global__ void amsgrad_sparse_update(float *param, const float *grad_data,
                                      const int *indices_data, float *m,
                                      float *v, float *maxv, float lr,
                                      float beta1, float beta2, float *betats,
                                      float eps, size_t size, size_t length) {
    size_t thread_ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_ind >= size)
        return;
    size_t ind = thread_ind / length;
    size_t offset = thread_ind % length;

    int grad_ind = indices_data[ind];
    if (grad_ind < 0)
        return;
    const float cur_grad = grad_data[thread_ind];
    size_t total_offset = length * grad_ind + offset;
    float *m_ptr = m + total_offset;
    float *v_ptr = v + total_offset;
    float *maxv_ptr = maxv + total_offset;
    float *param_ptr = param + total_offset;

    float cur_m = beta1 * (*m_ptr) + (1 - beta1) * cur_grad;
    float cur_v = beta2 * (*v_ptr) + (1 - beta2) * cur_grad * cur_grad;
    *m_ptr = cur_m;
    *v_ptr = cur_v;
    cur_m /= (1 - betats[0]);
    cur_v /= (1 - betats[1]);
    float cur_maxv = fmaxf(*maxv_ptr, cur_v);
    *maxv_ptr = cur_maxv;
    *(param_ptr) -= lr * cur_m / (sqrtf(cur_maxv) + eps);
}

int AdamOptimizerSparseUpdate(DLArrayHandle param,
                              const DLArrayHandle grad_indices,
                              const DLArrayHandle grad_values,
                              DLArrayHandle expavg, DLArrayHandle expavgsq,
                              DLArrayHandle maxv, float lr, float beta1,
                              float beta2, DLArrayHandle betats, float eps,
                              DLStreamHandle stream_handle = NULL) {
    size_t size = 1;
    size_t length = param->shape[1];
    for (int i = 0; i < grad_values->ndim; ++i) {
        size *= grad_values->shape[i];
    }

    dim3 blocks;
    dim3 threads;
    float *param_data = (float *)param->data;
    const float *grad_data = (const float *)grad_values->data;
    const int *indices_data = (const int *)grad_indices->data;
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
            amsgrad_sparse_update<<<blocks, threads, 0,
                                    *(cudaStream_t *)stream_handle->handle>>>(
                param_data, grad_data, indices_data, m_data, v_data, maxv_data,
                lr, beta1, beta2, betats_data, eps, size, length);
        } else {
            amsgrad_sparse_update<<<blocks, threads>>>(
                param_data, grad_data, indices_data, m_data, v_data, maxv_data,
                lr, beta1, beta2, betats_data, eps, size, length);
        }

    } else {
        if (stream_handle) {
            adam_sparse_update<<<blocks, threads, 0,
                                 *(cudaStream_t *)stream_handle->handle>>>(
                param_data, grad_data, indices_data, m_data, v_data, lr, beta1,
                beta2, betats_data, eps, size, length);
        } else {
            adam_sparse_update<<<blocks, threads>>>(
                param_data, grad_data, indices_data, m_data, v_data, lr, beta1,
                beta2, betats_data, eps, size, length);
        }
    }
    return 0;
}

__global__ void adamw_sparse_update(float *param, const float *grad_data,
                                    const int *indices_data, float *m, float *v,
                                    float lr, float beta1, float beta2,
                                    float *betats, float eps,
                                    float weight_decay, size_t size,
                                    size_t length) {
    size_t thread_ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_ind >= size)
        return;
    size_t ind = thread_ind / length;
    size_t offset = thread_ind % length;

    int grad_ind = indices_data[ind];
    if (grad_ind < 0)
        return;
    const float cur_grad = grad_data[thread_ind];
    size_t total_offset = length * grad_ind + offset;
    float *m_ptr = m + total_offset;
    float *v_ptr = v + total_offset;
    float *param_ptr = param + total_offset;

    float cur_m = beta1 * (*m_ptr) + (1 - beta1) * cur_grad;
    float cur_v = beta2 * (*v_ptr) + (1 - beta2) * cur_grad * cur_grad;
    *m_ptr = cur_m;
    *v_ptr = cur_v;
    cur_m /= (1 - betats[0]);
    cur_v /= (1 - betats[1]);
    float update = cur_m / (sqrtf(cur_v) + eps);
    *(param_ptr) -= lr * (update + weight_decay * (*param_ptr));
}

int AdamWOptimizerSparseUpdate(DLArrayHandle param,
                               const DLArrayHandle grad_indices,
                               const DLArrayHandle grad_values,
                               DLArrayHandle expavg, DLArrayHandle expavgsq,
                               float lr, float beta1, float beta2,
                               DLArrayHandle betats, float eps,
                               float weight_decay,
                               DLStreamHandle stream_handle = NULL) {
    size_t size = 1;
    size_t length = param->shape[1];
    for (int i = 0; i < grad_values->ndim; ++i) {
        size *= grad_values->shape[i];
    }

    dim3 blocks;
    dim3 threads;
    float *param_data = (float *)param->data;
    const float *grad_data = (const float *)grad_values->data;
    const int *indices_data = (const int *)grad_indices->data;
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
        adamw_sparse_update<<<blocks, threads, 0,
                              *(cudaStream_t *)stream_handle->handle>>>(
            param_data, grad_data, indices_data, m_data, v_data, lr, beta1,
            beta2, betats_data, eps, weight_decay, size, length);
    else
        adamw_sparse_update<<<blocks, threads>>>(
            param_data, grad_data, indices_data, m_data, v_data, lr, beta1,
            beta2, betats_data, eps, weight_decay, size, length);
    return 0;
}

__global__ void get_indexed_params(float *indexed_param, const float *param,
                                   const int *indices_data, size_t size,
                                   size_t length) {
    size_t thread_ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_ind >= size)
        return;
    size_t ind = thread_ind / length;
    size_t offset = thread_ind % length;

    int grad_ind = indices_data[ind];
    if (grad_ind < 0)
        return;
    size_t total_offset = length * grad_ind + offset;
    const float *param_ptr = param + total_offset;
    indexed_param[thread_ind] = *(param_ptr);
}

__global__ void calc_lamb_update_sparse(float *update, const float *grad_data,
                                        const int *indices_data, float *m,
                                        float *v, float beta1, float beta2,
                                        float *betats, float eps, size_t size,
                                        size_t length) {
    size_t thread_ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_ind >= size)
        return;
    size_t ind = thread_ind / length;
    size_t offset = thread_ind % length;

    int grad_ind = indices_data[ind];
    if (grad_ind < 0)
        return;
    const float cur_grad = grad_data[thread_ind];
    size_t total_offset = length * grad_ind + offset;
    float *m_ptr = m + total_offset;
    float *v_ptr = v + total_offset;

    float cur_m = beta1 * (*m_ptr) + (1 - beta1) * cur_grad;
    float cur_v = beta2 * (*v_ptr) + (1 - beta2) * cur_grad * cur_grad;
    *m_ptr = cur_m;
    *v_ptr = cur_v;
    cur_m /= (1 - betats[0]);
    cur_v /= (1 - betats[1]);
    update[thread_ind] = cur_m / (sqrtf(cur_v) + eps);
}

__global__ void lamb_update_step_sparse(float *param, const float *update,
                                        const int *indices_data, float lr,
                                        float weight_decay, float *norm2_param,
                                        float *norm2_update, size_t size,
                                        size_t length) {
    size_t thread_ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_ind >= size)
        return;
    size_t ind = thread_ind / length;
    size_t offset = thread_ind % length;

    int grad_ind = indices_data[ind];
    if (grad_ind < 0)
        return;
    const float cur_update = update[thread_ind];
    size_t total_offset = length * grad_ind + offset;
    float *param_ptr = param + total_offset;
    *(param_ptr) -= lr * (norm2_param[0] / norm2_update[0])
                    * (cur_update + weight_decay * (*param_ptr));
}

int LambOptimizerSparseUpdate(DLArrayHandle param,
                              const DLArrayHandle grad_indices,
                              const DLArrayHandle grad_values,
                              DLArrayHandle expavg, DLArrayHandle expavgsq,
                              float lr, float beta1, float beta2,
                              DLArrayHandle betats, float eps,
                              float weight_decay,
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
    int *dim_sparse = (int *)malloc(cpu_mem);
    int *stride_sparse = (int *)malloc(cpu_mem);
    int *dimC = (int *)malloc(cpu_mem);
    int *strideC = (int *)malloc(cpu_mem);

    size_t size_sparse = 1;
    size_t length = param->shape[1];
    for (int i = 0; i < grad_values->ndim; ++i) {
        size_sparse *= grad_values->shape[i];
    }

    int temp_stride_sparse = 1;
    int temp_strideC = 1;

    for (int i = ndim - 1; i >= 0; --i) {
        dim_sparse[i] = i < grad_values->ndim ? (int)grad_values->shape[i] : 1;
        dimC[i] = 1;
        stride_sparse[i] = temp_stride_sparse;
        strideC[i] = temp_strideC;
        temp_stride_sparse *= dim_sparse[i];
        temp_strideC *= dimC[i];
    }
    size_t byte_size_sparse = temp_stride_sparse * sizeof(float);

    dim3 blocks;
    dim3 threads;
    float *param_data = (float *)param->data;
    const float *grad_data = (const float *)grad_values->data;
    const int *indices_data = (const int *)grad_indices->data;
    float *m_data = (float *)expavg->data;
    float *v_data = (float *)expavgsq->data;
    float *betats_data = (float *)betats->data;
    if (size_sparse <= 1024) {
        threads.x = size_sparse;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size_sparse + 1023) / 1024;
    }

    if (is_chunk_init(dev_id) == false) {
        chunk_init(dev_id);
    }
    void *norm2_param = find_chunk(1 * sizeof(float), dev_id);
    void *norm2_update = find_chunk(1 * sizeof(float), dev_id);
    void *workspace = find_chunk(byte_size_sparse, dev_id);
    void *intermediate = find_chunk(byte_size_sparse, dev_id);

    // Get indexed params to intermediate for Norm2
    if (stream_handle)
        get_indexed_params<<<blocks, threads, 0,
                             *(cudaStream_t *)stream_handle->handle>>>(
            (float *)intermediate, (const float *)param_data, indices_data,
            size_sparse, length);
    else
        get_indexed_params<<<blocks, threads>>>(
            (float *)intermediate, (const float *)param_data, indices_data,
            size_sparse, length);

    // Calculate Norm2 of param indexed
    CUDNN_CALL(cudnnSetTensorNdDescriptor(adesc, CUDNN_DATA_FLOAT, ndim,
                                          dim_sparse, stride_sparse));
    CUDNN_CALL(cudnnSetTensorNdDescriptor(cdesc, CUDNN_DATA_FLOAT, ndim, dimC,
                                          strideC));
    CUDNN_CALL(cudnnReduceTensor(
        cudnn_map[dev_id], rtd, NULL, 0, workspace, byte_size_sparse, &one,
        adesc, (const void *)intermediate, &zero, cdesc, norm2_param));

    // Calculate update
    if (stream_handle)
        calc_lamb_update_sparse<<<blocks, threads, 0,
                                  *(cudaStream_t *)stream_handle->handle>>>(
            (float *)intermediate, grad_data, indices_data, m_data, v_data,
            beta1, beta2, betats_data, eps, size_sparse, length);
    else
        calc_lamb_update_sparse<<<blocks, threads>>>(
            (float *)intermediate, grad_data, indices_data, m_data, v_data,
            beta1, beta2, betats_data, eps, size_sparse, length);

    // Calculate Norm2 of update
    CUDNN_CALL(cudnnSetTensorNdDescriptor(adesc, CUDNN_DATA_FLOAT, ndim,
                                          dim_sparse, stride_sparse));
    CUDNN_CALL(cudnnSetTensorNdDescriptor(cdesc, CUDNN_DATA_FLOAT, ndim, dimC,
                                          strideC));
    CUDNN_CALL(cudnnReduceTensor(
        cudnn_map[dev_id], rtd, NULL, 0, workspace, byte_size_sparse, &one,
        adesc, (const void *)intermediate, &zero, cdesc, norm2_update));

    // Update step
    if (stream_handle)
        lamb_update_step_sparse<<<blocks, threads, 0,
                                  *(cudaStream_t *)stream_handle->handle>>>(
            param_data, (const float *)intermediate, indices_data, lr,
            weight_decay, (float *)norm2_param, (float *)norm2_update,
            size_sparse, length);
    else
        lamb_update_step_sparse<<<blocks, threads>>>(
            param_data, (const float *)intermediate, indices_data, lr,
            weight_decay, (float *)norm2_param, (float *)norm2_update,
            size_sparse, length);

    del_chunk(norm2_param, dev_id);
    del_chunk(norm2_update, dev_id);
    del_chunk(workspace, dev_id);
    del_chunk(intermediate, dev_id);
    CUDNN_CALL(cudnnDestroyTensorDescriptor(adesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(cdesc));
    CUDNN_CALL(cudnnDestroyReduceTensorDescriptor(rtd));
    free(dim_sparse);
    free(dimC);
    free(stride_sparse);
    free(strideC);
    return 0;
}
