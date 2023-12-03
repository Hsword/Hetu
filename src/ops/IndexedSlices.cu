#include "gpu_runtime.h"

__global__ void indexedslices_oneside_add_kernel(const float *values_data,
                                                 const int *indices_data,
                                                 float *output_data,
                                                 size_t size, size_t nrow,
                                                 size_t length) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size)
        return;
    int id = indices_data[index];
    if (id < 0 || id >= nrow)
        return;
    const float *values_ptr = values_data + length * index;
    float *output_ptr = output_data + length * id;
    for (int i = 0; i < length; i++)
        output_ptr[i] += values_ptr[i];
}

int IndexedSlicesOneSideAdd(const DLArrayHandle indices,
                            const DLArrayHandle values, DLArrayHandle output,
                            DLStreamHandle stream_handle = NULL) {
    size_t size = ArrSize(indices);
    assert(output->ndim == 2);
    size_t nrow = output->shape[0];
    size_t length = output->shape[1];

    dim3 blocks;
    dim3 threads;
    ThreadBlock1D(threads, blocks, size);
    const float *values_data = (const float *)values->data;
    float *output_data = (float *)output->data;
    const int *indices_data = (const int *)indices->data;

    if (stream_handle)
        indexedslices_oneside_add_kernel<<<
            blocks, threads, 0, *(cudaStream_t *)stream_handle->handle>>>(
            values_data, indices_data, output_data, size, nrow, length);
    else
        indexedslices_oneside_add_kernel<<<blocks, threads>>>(
            values_data, indices_data, output_data, size, nrow, length);
    return 0;
}
