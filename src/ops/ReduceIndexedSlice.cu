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
    if (ind >= unique_size)
        return;
    int l = id_length[ind], r = id_length[ind + 1];
    float sum = 0;
    for (int i = l; i < r; ++i) {
        int offset = id_offset[i];
        sum += in_value[offset * width + wid];
    }
    out_value[ind * width + wid] = sum;
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