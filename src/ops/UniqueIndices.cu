#include "gpu_runtime.h"

#include <cmath>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

int DLGpuGetUniqueWorkspaceSize(size_t ind_size, size_t *size) {
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

extern __global__ void range_kernel(int *array, size_t size);

extern __global__ void set_invalid_kernel(int *indices, int *punique_size,
                                          size_t size);

int DLGpuUniqueIndices(const DLArrayHandle indices, DLArrayHandle output,
                       DLArrayHandle idoffsets, DLArrayHandle workspace,
                       size_t storage_size, int end_bit,
                       DLStreamHandle stream_handle = NULL) {
    size_t ind_size = ArrSize(indices);
    int *punique_size = (int *)(idoffsets->data);
    int *id_offset = punique_size + 1;
    int *id_length = id_offset + ind_size;
    void *temp_data = workspace->data;
    const int *in_ind_data = (const int *)indices->data;
    int *out_ind_data = (int *)output->data;

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
    if (stream_handle) {
        set_invalid_kernel<<<blocks, threads, 0, stream>>>(
            out_ind_data, punique_size, ind_size);
    } else {
        set_invalid_kernel<<<blocks, threads>>>(out_ind_data, punique_size,
                                                ind_size);
    }
    return 0;
}

__global__ void dedup_lookup_kernel(const float *in_value, float *out_value,
                                    const int *id_length, const int *id_offset,
                                    const int *punique_size, size_t width,
                                    size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
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

int DLGpuDedupLookup(const DLArrayHandle lookups, const DLArrayHandle idoffsets,
                     DLArrayHandle output,
                     DLStreamHandle stream_handle = NULL) {
    size_t ind_size = 1;
    for (int i = 0; i < lookups->ndim - 1; ++i) {
        ind_size *= lookups->shape[i];
    }
    size_t width = lookups->shape[lookups->ndim - 1];
    size_t size = ind_size * width;
    const int *punique_size = (const int *)idoffsets->data;
    const int *id_offset = punique_size + 1;
    const int *id_length = id_offset + ind_size;
    const float *in_val_data = (const float *)lookups->data;
    float *out_val_data = (float *)output->data;

    dim3 threads, blocks;
    ThreadBlock1D(threads, blocks, size);
    if (stream_handle) {
        dedup_lookup_kernel<<<blocks, threads, 0,
                              *(cudaStream_t *)stream_handle->handle>>>(
            in_val_data, out_val_data, id_length, id_offset, punique_size,
            width, size);
    } else {
        dedup_lookup_kernel<<<blocks, threads>>>(in_val_data, out_val_data,
                                                 id_length, id_offset,
                                                 punique_size, width, size);
    }
    return 0;
}

__global__ void dedup_grad_kernel(const float *in_value, float *out_value,
                                  const int *id_length, const int *id_offset,
                                  const int *punique_size, size_t width,
                                  size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
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

int DLGpuDedupGrad(const DLArrayHandle grad, const DLArrayHandle idoffsets,
                   DLArrayHandle output, DLStreamHandle stream_handle = NULL) {
    size_t ind_size = 1;
    for (int i = 0; i < grad->ndim - 1; ++i) {
        ind_size *= grad->shape[i];
    }
    size_t width = grad->shape[grad->ndim - 1];
    size_t size = ind_size * width;
    const int *punique_size = (const int *)idoffsets->data;
    const int *id_offset = punique_size + 1;
    const int *id_length = id_offset + ind_size;
    const float *in_val_data = (const float *)grad->data;
    float *out_val_data = (float *)output->data;

    dim3 threads, blocks;
    ThreadBlock1D(threads, blocks, size);
    if (stream_handle) {
        dedup_grad_kernel<<<blocks, threads, 0,
                            *(cudaStream_t *)stream_handle->handle>>>(
            in_val_data, out_val_data, id_length, id_offset, punique_size,
            width, size);
    } else {
        dedup_grad_kernel<<<blocks, threads>>>(in_val_data, out_val_data,
                                               id_length, id_offset,
                                               punique_size, width, size);
    }
    return 0;
}
