#include "gpu_runtime.h"

__global__ void sgd_sparse_update(const float *grad_data,
                                  const float *indices_data, float *param_data,
                                  size_t size, size_t length, float lr) {
    size_t thread_ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_ind >= size)
        return;
    size_t index = thread_ind / length;
    size_t offset = thread_ind % length;
    int id = indices_data[index];
    const float cur_grad = grad_data[thread_ind];
    float *param_ptr = param_data + length * id + offset;
    atomicAdd(param_ptr, -lr * cur_grad);
}

int SGDOptimizerSparseUpdate(DLArrayHandle param,
                             const DLArrayHandle grad_indices,
                             const DLArrayHandle grad_values, float lr,
                             DLStreamHandle stream_handle = NULL) {
    size_t size = 1;
    size_t length = param->shape[1];
    for (int i = 0; i < grad_values->ndim; i++) {
        size *= grad_values->shape[i];
    }

    dim3 blocks;
    dim3 threads;
    const float *grad_data = (const float *)grad_values->data;
    float *param_data = (float *)param->data;
    const float *indices_data = (const float *)grad_indices->data;

    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }

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
                                            const float *indices_data, float lr,
                                            size_t size, size_t length) {
    size_t thread_ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_ind >= size)
        return;
    size_t ind = thread_ind / length;
    size_t offset = thread_ind % length;
    int grad_ind = indices_data[ind];
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
                                            const float *indices_data, float lr,
                                            size_t size, size_t length) {
    size_t thread_ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_ind >= size)
        return;
    size_t ind = thread_ind / length;
    size_t offset = thread_ind % length;
    int grad_ind = indices_data[ind];
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
    const float *indices_data = (const float *)grad_indices->data;
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
                                      const float *indices_data,
                                      float *acc_data, float lr, float eps,
                                      size_t size, size_t length) {
    size_t thread_ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_ind >= size)
        return;
    size_t ind = thread_ind / length;
    size_t offset = thread_ind % length;

    int grad_ind = indices_data[ind];
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
    size_t size = 1;
    size_t length = param->shape[1];
    for (int i = 0; i < grad_values->ndim; ++i) {
        size *= grad_values->shape[i];
    }

    float *param_data = (float *)param->data;
    const float *grad_data = (const float *)grad_values->data;
    const float *indices_data = (const float *)grad_indices->data;
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
                                   const float *indices_data, float *m,
                                   float *v, float lr, float beta1, float beta2,
                                   float beta1t, float beta2t, float eps,
                                   size_t size, size_t length) {
    size_t thread_ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_ind >= size)
        return;
    size_t ind = thread_ind / length;
    size_t offset = thread_ind % length;

    int grad_ind = indices_data[ind];
    const float cur_grad = grad_data[thread_ind];
    size_t total_offset = length * grad_ind + offset;
    float *m_ptr = m + total_offset;
    float *v_ptr = v + total_offset;
    float *param_ptr = param + total_offset;

    float cur_m = beta1 * (*m_ptr) + (1 - beta1) * cur_grad;
    float cur_v = beta2 * (*v_ptr) + (1 - beta2) * cur_grad * cur_grad;
    *m_ptr = cur_m;
    *v_ptr = cur_v;
    cur_m /= (1 - beta1t);
    cur_v /= (1 - beta2t);
    *(param_ptr) -= lr * cur_m / (sqrtf(cur_v) + eps);
}

int AdamOptimizerSparseUpdate(DLArrayHandle param,
                              const DLArrayHandle grad_indices,
                              const DLArrayHandle grad_values,
                              DLArrayHandle expavg, DLArrayHandle expavgsq,
                              float lr, float beta1, float beta2, float beta1t,
                              float beta2t, float eps,
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
    const float *indices_data = (const float *)grad_indices->data;
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
        adam_sparse_update<<<blocks, threads, 0,
                             *(cudaStream_t *)stream_handle->handle>>>(
            param_data, grad_data, indices_data, m_data, v_data, lr, beta1,
            beta2, beta1t, beta2t, eps, size, length);
    else
        adam_sparse_update<<<blocks, threads>>>(
            param_data, grad_data, indices_data, m_data, v_data, lr, beta1,
            beta2, beta1t, beta2t, eps, size, length);
    return 0;
}
