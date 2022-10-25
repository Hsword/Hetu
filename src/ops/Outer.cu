#include "gpu_runtime.h"

__global__ void outer_kernel(const float *input, const float *vector,
                             float *output, size_t ncol, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    size_t row_id = ind / ncol;
    size_t col_id = ind % ncol;
    output[ind] = input[row_id] * vector[col_id];
}

int DLGpuOuter(const DLArrayHandle input, const DLArrayHandle vector,
               DLArrayHandle output, DLStreamHandle stream_handle = NULL) {
    size_t size = 1;
    for (index_t i = 0; i < output->ndim; i++) {
        size *= output->shape[i];
    }
    size_t ncol = output->shape[1];
    dim3 blocks;
    dim3 threads;
    const float *input_data = (const float *)input->data;
    const float *vector_data = (const float *)vector->data;
    float *output_data = (float *)output->data;
    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }
    if (stream_handle)
        outer_kernel<<<blocks, threads, 0,
                       *(cudaStream_t *)stream_handle->handle>>>(
            input_data, vector_data, output_data, ncol, size);
    else
        outer_kernel<<<blocks, threads>>>(input_data, vector_data, output_data,
                                          ncol, size);
    return 0;
}
