#include "gpu_runtime.h"

__global__ void sparse_set(const int *data, const int *indices, int *table,
                           size_t size, size_t length) {
    size_t thread_ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_ind >= size)
        return;
    size_t index = thread_ind / length;
    size_t offset = thread_ind % length;
    int id = indices[index];
    if (id < 0)
        return;
    const int cur_data = data[thread_ind];
    int *table_ptr = table + length * id + offset;
    atomicExch(table_ptr, cur_data);
}

int DLGpuSparseSet(DLArrayHandle table, const DLArrayHandle indices,
                   const DLArrayHandle data,
                   DLStreamHandle stream_handle = NULL) {
    size_t size = ArrSize(data);
    size_t length = table->shape[1];

    dim3 blocks;
    dim3 threads;
    ThreadBlock1D(threads, blocks, size);
    const int *datavalue = (const int *)data->data;
    int *tablevalue = (int *)table->data;
    const int *indvalue = (const int *)indices->data;

    if (stream_handle)
        sparse_set<<<blocks, threads, 0,
                     *(cudaStream_t *)stream_handle->handle>>>(
            datavalue, indvalue, tablevalue, size, length);
    else
        sparse_set<<<blocks, threads>>>(datavalue, indvalue, tablevalue, size,
                                        length);
    return 0;
}
