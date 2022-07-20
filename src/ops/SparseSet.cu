#include "gpu_runtime.h"

__global__ void sparse_set(const float *data, const int *indices, float *table,
                           size_t size, size_t length) {
    size_t thread_ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_ind >= size)
        return;
    size_t index = thread_ind / length;
    size_t offset = thread_ind % length;
    int id = indices[index];
    if (id < 0)
        return;
    const float cur_data = data[thread_ind];
    float *table_ptr = table + length * id + offset;
    atomicExch(table_ptr, cur_data);
}

int DLGpuSparseSet(DLArrayHandle table, const DLArrayHandle indices,
                   const DLArrayHandle data,
                   DLStreamHandle stream_handle = NULL) {
    size_t size = 1;
    size_t length = table->shape[1];
    for (int i = 0; i < data->ndim; i++) {
        size *= data->shape[i];
    }

    dim3 blocks;
    dim3 threads;
    const float *datavalue = (const float *)data->data;
    float *tablevalue = (float *)table->data;
    const int *indvalue = (const int *)indices->data;

    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }

    if (stream_handle)
        sparse_set<<<blocks, threads, 0,
                     *(cudaStream_t *)stream_handle->handle>>>(
            datavalue, indvalue, tablevalue, size, length);
    else
        sparse_set<<<blocks, threads>>>(datavalue, indvalue, tablevalue, size,
                                        length);
    return 0;
}
