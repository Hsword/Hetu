#include "gpu_runtime.h"

#include <cmath>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

__global__ void range_kernel(int *array, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    array[ind] = ind;
}

__global__ void reduce_indexedslices_kernel(const float *in_value,
                                            float *out_value, int *id_length,
                                            int *id_offset, int *punique_size,
                                            size_t width) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    size_t wid = ind % width;
    ind /= width;
    int unique_size = *punique_size;
    if (ind < unique_size) {
        int l = id_length[ind], r = id_length[ind + 1];
        float sum = 0;
        for (int i = l; i < r; ++i) {
            int offset = id_offset[i];
            sum += in_value[offset * width + wid];
        }
        out_value[ind * width + wid] = sum;
    } else {
        out_value[ind * width + wid] = 0;
    }
}

__global__ void unique_indexedslices_kernel(const float *in_value,
                                            float *out_value, int *id_length,
                                            int *id_offset, int *punique_size,
                                            size_t width) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    size_t wid = ind % width;
    ind /= width;
    int unique_size = *punique_size;
    if (ind < unique_size) {
        int l = id_length[ind];
        out_value[ind * width + wid] = in_value[id_offset[l] * width + wid];
    } else {
        out_value[ind * width + wid] = 0;
    }
}

__global__ void set_invalid_kernel(int *indices, int *punique_size,
                                   size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    int unique_size = *punique_size;
    if (ind < unique_size || ind >= size)
        return;
    indices[ind] = -1;
}

int DLGpuReduceIndexedSlice(const DLArrayHandle in_indices,
                            const DLArrayHandle in_values,
                            DLArrayHandle out_indices, DLArrayHandle out_values,
                            DLArrayHandle workspace, size_t storage_size,
                            int end_bit, DLStreamHandle stream_handle = NULL) {
    size_t ind_size = ArrSize(in_indices);
    size_t width = in_values->shape[in_values->ndim - 1];
    int *int_temp = (int *)(workspace->data);
    int *id_offset = int_temp;
    int *id_length = int_temp + ind_size;
    int *punique_size = id_length + ind_size + 1;
    void *temp_data = (void *)(punique_size + 1);
    const int *in_ind_data = (const int *)in_indices->data;
    const float *in_val_data = (const float *)in_values->data;
    int *out_ind_data = (int *)out_indices->data;
    float *out_val_data = (float *)out_values->data;

    dim3 threads, blocks;
    ThreadBlock1D(threads, blocks, ind_size);
    cudaStream_t stream = NULL;
    if (stream_handle) {
        stream = *(cudaStream_t *)stream_handle->handle;
        range_kernel<<<blocks, threads, 0, stream>>>(id_length, ind_size);
    } else {
        range_kernel<<<blocks, threads>>>(id_length, ind_size);
    }
    CUDA_CALL(cub::DeviceRadixSort::SortPairs(
        temp_data, storage_size, in_ind_data, out_ind_data, id_length,
        id_offset, ind_size, 0, end_bit, stream));
    CUDA_CALL(cub::DeviceRunLengthEncode::Encode(
        temp_data, storage_size, out_ind_data, out_ind_data, id_length,
        punique_size, ind_size, stream));
    CUDA_CALL(cub::DeviceScan::ExclusiveSum(temp_data, storage_size, id_length,
                                            id_length, ind_size + 1, stream));
    ThreadBlock1D(threads, blocks, ind_size * width);
    if (stream_handle) {
        reduce_indexedslices_kernel<<<blocks, threads, 0, stream>>>(
            in_val_data, out_val_data, id_length, id_offset, punique_size,
            width);
    } else {
        reduce_indexedslices_kernel<<<blocks, threads>>>(
            in_val_data, out_val_data, id_length, id_offset, punique_size,
            width);
    }
    ThreadBlock1D(threads, blocks, ind_size);
    if (stream_handle) {
        set_invalid_kernel<<<blocks, threads, 0, stream>>>(
            out_ind_data, punique_size, ind_size);
    } else {
        set_invalid_kernel<<<blocks, threads>>>(out_ind_data, punique_size,
                                                ind_size);
    }
    return 0;
}

int DLGpuReduceIndexedSliceGetWorkspaceSize(size_t ind_size, size_t *size) {
    size_t max_size = 0;
    size_t cur_size = 0;
    int *ptr = nullptr;
    CUDA_CALL(cub::DeviceRadixSort::SortPairs(nullptr, cur_size, ptr, ptr, ptr,
                                              ptr, ind_size));
    if (cur_size > max_size)
        max_size = cur_size;
    CUDA_CALL(cub::DeviceRunLengthEncode::Encode(nullptr, cur_size, ptr, ptr,
                                                 ptr, ptr, ind_size));
    if (cur_size > max_size)
        max_size = cur_size;
    CUDA_CALL(cub::DeviceScan::ExclusiveSum(nullptr, cur_size, ptr, ptr,
                                            ind_size + 1));
    if (cur_size > max_size)
        max_size = cur_size;
    *size = max_size;
    return 0;
}

int DLGpuReduceIndexedSliceWithEmbedding(
    const DLArrayHandle in_indices, const DLArrayHandle in_values,
    const DLArrayHandle parameters, DLArrayHandle out_indices,
    DLArrayHandle out_values, DLArrayHandle out_params, DLArrayHandle workspace,
    size_t storage_size, int end_bit, DLStreamHandle stream_handle = NULL) {
    size_t ind_size = ArrSize(in_indices);
    size_t width = in_values->shape[in_values->ndim - 1];
    int *int_temp = (int *)(workspace->data);
    int *id_offset = int_temp;
    int *id_length = int_temp + ind_size;
    int *punique_size = id_length + ind_size + 1;
    void *temp_data = (void *)(punique_size + 1);
    const int *in_ind_data = (const int *)in_indices->data;
    const float *in_val_data = (const float *)in_values->data;
    const float *in_par_data = (const float *)parameters->data;
    int *out_ind_data = (int *)out_indices->data;
    float *out_val_data = (float *)out_values->data;
    float *out_par_data = (float *)out_params->data;

    dim3 threads, blocks;
    ThreadBlock1D(threads, blocks, ind_size);
    cudaStream_t stream = NULL;
    if (stream_handle) {
        stream = *(cudaStream_t *)stream_handle->handle;
        range_kernel<<<blocks, threads, 0, stream>>>(id_length, ind_size);
    } else {
        range_kernel<<<blocks, threads>>>(id_length, ind_size);
    }
    CUDA_CALL(cub::DeviceRadixSort::SortPairs(
        temp_data, storage_size, in_ind_data, out_ind_data, id_length,
        id_offset, ind_size, 0, end_bit, stream));
    CUDA_CALL(cub::DeviceRunLengthEncode::Encode(
        temp_data, storage_size, out_ind_data, out_ind_data, id_length,
        punique_size, ind_size, stream));
    CUDA_CALL(cub::DeviceScan::ExclusiveSum(temp_data, storage_size, id_length,
                                            id_length, ind_size + 1, stream));
    ThreadBlock1D(threads, blocks, ind_size * width);
    if (stream_handle) {
        reduce_indexedslices_kernel<<<blocks, threads, 0, stream>>>(
            in_val_data, out_val_data, id_length, id_offset, punique_size,
            width);
        unique_indexedslices_kernel<<<blocks, threads, 0, stream>>>(
            in_par_data, out_par_data, id_length, id_offset, punique_size,
            width);
    } else {
        reduce_indexedslices_kernel<<<blocks, threads>>>(
            in_val_data, out_val_data, id_length, id_offset, punique_size,
            width);
        unique_indexedslices_kernel<<<blocks, threads>>>(
            in_par_data, out_par_data, id_length, id_offset, punique_size,
            width);
    }
    ThreadBlock1D(threads, blocks, ind_size);
    if (stream_handle) {
        set_invalid_kernel<<<blocks, threads, 0, stream>>>(
            out_ind_data, punique_size, ind_size);
    } else {
        set_invalid_kernel<<<blocks, threads>>>(out_ind_data, punique_size,
                                                ind_size);
    }
    return 0;
}

__global__ void sgd_update_indexedslices_kernel(const int *indices,
                                                const float *grads,
                                                const float *params,
                                                float *output, float lr,
                                                size_t dim, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    int index = indices[ind / dim];
    if (index < 0)
        return;
    output[ind] = params[ind] - lr * grads[ind];
}

int DLGpuSGDUpdateIndexedSlices(const DLArrayHandle indices,
                                const DLArrayHandle grads,
                                const DLArrayHandle params,
                                DLArrayHandle output, float lr,
                                DLStreamHandle stream_handle = NULL) {
    size_t size = ArrSize(grads);
    size_t dim = grads->shape[grads->ndim - 1];
    const int *indices_data = (const int *)indices->data;
    const float *grads_data = (const float *)grads->data;
    const float *params_data = (const float *)params->data;
    float *output_data = (float *)output->data;
    dim3 threads, blocks;
    ThreadBlock1D(threads, blocks, size);
    if (stream_handle) {
        sgd_update_indexedslices_kernel<<<
            blocks, threads, 0, *(cudaStream_t *)stream_handle->handle>>>(
            indices_data, grads_data, params_data, output_data, lr, dim, size);
    } else {
        sgd_update_indexedslices_kernel<<<blocks, threads>>>(
            indices_data, grads_data, params_data, output_data, lr, dim, size);
    }
    return 0;
}

__global__ void adagrad_update_indexedslices_kernel(
    const int *indices, const float *grads, const float *params, float *output,
    float lr, float *accum, float eps, size_t dim, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    int index = indices[ind / dim];
    if (index < 0)
        return;
    size_t offset = (size_t)index * dim + ind % dim;
    accum[offset] = accum[offset] + grads[ind] * grads[ind];
    output[ind] = params[ind] - lr * grads[ind] / (sqrtf(accum[offset]) + eps);
}

int DLGpuAdaGradUpdateIndexedSlices(const DLArrayHandle indices,
                                    const DLArrayHandle grads,
                                    const DLArrayHandle params,
                                    DLArrayHandle output, float lr,
                                    DLArrayHandle accum, float epsilon,
                                    DLStreamHandle stream_handle = NULL) {
    size_t size = ArrSize(grads);
    size_t dim = grads->shape[grads->ndim - 1];
    const int *indices_data = (const int *)indices->data;
    const float *grads_data = (const float *)grads->data;
    const float *params_data = (const float *)params->data;
    float *output_data = (float *)output->data;
    float *accum_data = (float *)accum->data;
    dim3 threads, blocks;
    ThreadBlock1D(threads, blocks, size);
    if (stream_handle) {
        adagrad_update_indexedslices_kernel<<<
            blocks, threads, 0, *(cudaStream_t *)stream_handle->handle>>>(
            indices_data, grads_data, params_data, output_data, lr, accum_data,
            epsilon, dim, size);
    } else {
        adagrad_update_indexedslices_kernel<<<blocks, threads>>>(
            indices_data, grads_data, params_data, output_data, lr, accum_data,
            epsilon, dim, size);
    }
    return 0;
}

__global__ void adam_update_indexedslices_kernel(
    const int *indices, const float *grads, const float *params, float *output,
    float lr, float *m, float *v, const float *betats, float beta1, float beta2,
    float eps, size_t dim, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    int index = indices[ind / dim];
    if (index < 0)
        return;
    size_t offset = (size_t)index * dim + ind % dim;
    m[offset] = beta1 * m[offset] + (1 - beta1) * grads[ind];
    v[offset] = beta2 * v[offset] + (1 - beta2) * grads[ind] * grads[ind];
    float m_local = m[offset] / (1 - betats[0]);
    float v_local = v[offset] / (1 - betats[1]);
    output[ind] = params[ind] - lr * m_local / (sqrtf(v_local) + eps);
}

__global__ void amsgrad_update_indexedslices_kernel(
    const int *indices, const float *grads, const float *params, float *output,
    float lr, float *m, float *v, float *maxv, const float *betats, float beta1,
    float beta2, float eps, size_t dim, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    int index = indices[ind / dim];
    if (index < 0)
        return;
    size_t offset = (size_t)index * dim + ind % dim;
    m[offset] = beta1 * m[offset] + (1 - beta1) * grads[ind];
    v[offset] = beta2 * v[offset] + (1 - beta2) * grads[ind] * grads[ind];
    float m_local = m[offset] / (1 - betats[0]);
    float v_local = v[offset] / (1 - betats[1]);
    float cur_maxv = fmaxf(v_local, maxv[offset]);
    maxv[offset] = cur_maxv;
    output[ind] = params[ind] - lr * m_local / (sqrtf(cur_maxv) + eps);
}

int DLGpuAdamUpdateIndexedSlices(
    const DLArrayHandle indices, const DLArrayHandle grads,
    const DLArrayHandle params, DLArrayHandle output, float lr, DLArrayHandle m,
    DLArrayHandle v, DLArrayHandle maxv, float beta1, float beta2,
    DLArrayHandle betats, float epsilon, DLStreamHandle stream_handle = NULL) {
    size_t size = ArrSize(grads);
    size_t dim = grads->shape[grads->ndim - 1];
    const int *indices_data = (const int *)indices->data;
    const float *grads_data = (const float *)grads->data;
    const float *params_data = (const float *)params->data;
    float *output_data = (float *)output->data;
    float *m_data = (float *)m->data;
    float *v_data = (float *)v->data;
    const float *betats_data = (const float *)betats->data;
    dim3 threads, blocks;
    ThreadBlock1D(threads, blocks, size);
    if (maxv == NULL) {
        if (stream_handle) {
            adam_update_indexedslices_kernel<<<
                blocks, threads, 0, *(cudaStream_t *)stream_handle->handle>>>(
                indices_data, grads_data, params_data, output_data, lr, m_data,
                v_data, betats_data, beta1, beta2, epsilon, dim, size);
        } else {
            adam_update_indexedslices_kernel<<<blocks, threads>>>(
                indices_data, grads_data, params_data, output_data, lr, m_data,
                v_data, betats_data, beta1, beta2, epsilon, dim, size);
        }
    } else {
        float *maxv_data = (float *)maxv->data;
        if (stream_handle) {
            amsgrad_update_indexedslices_kernel<<<
                blocks, threads, 0, *(cudaStream_t *)stream_handle->handle>>>(
                indices_data, grads_data, params_data, output_data, lr, m_data,
                v_data, maxv_data, betats_data, beta1, beta2, epsilon, dim,
                size);
        } else {
            amsgrad_update_indexedslices_kernel<<<blocks, threads>>>(
                indices_data, grads_data, params_data, output_data, lr, m_data,
                v_data, maxv_data, betats_data, beta1, beta2, epsilon, dim,
                size);
        }
    }
    return 0;
}
