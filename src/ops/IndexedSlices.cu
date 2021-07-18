#include "gpu_runtime.h"

__global__ void indexedslices_oneside_add_kernel(const float *values_data,
                                                 const float *indices_data,
                                                 float *output_data,
                                                 size_t size, size_t length) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size)
        return;
    int id = indices_data[index];
    const float *values_ptr = values_data + length * index;
    float *output_ptr = output_data + length * id;
    for (int i = 0; i < length; i++)
        atomicAdd(output_ptr + i, *(values_ptr + i));
}

int IndexedSlicesOneSideAdd(const DLArrayHandle indices,
                            const DLArrayHandle values, DLArrayHandle output,
                            DLStreamHandle stream_handle = NULL) {
    size_t size = 1;
    size_t length = output->shape[1];
    for (int i = 0; i < indices->ndim; i++) {
        size *= indices->shape[i];
    }

    dim3 blocks;
    dim3 threads;
    const float *values_data = (const float *)values->data;
    float *output_data = (float *)output->data;
    const float *indices_data = (const float *)indices->data;

    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }

    if (stream_handle)
        indexedslices_oneside_add_kernel<<<
            blocks, threads, 0, *(cudaStream_t *)stream_handle->handle>>>(
            values_data, indices_data, output_data, size, length);
    else
        indexedslices_oneside_add_kernel<<<blocks, threads>>>(
            values_data, indices_data, output_data, size, length);
    return 0;
}
