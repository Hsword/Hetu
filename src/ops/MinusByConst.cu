#include "gpu_runtime.h"

__global__ void minus_by_const_kernel(const float *input, float *output,
                                      float val, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size) {
        return;
    }
    output[ind] = val - input[ind];
}

int DLGpuMinusByConst(const DLArrayHandle input, DLArrayHandle output,
                      float val, DLStreamHandle stream_handle = NULL) {
    size_t size = 1;
    for (int i = 0; i < output->ndim; i++) {
        size *= output->shape[i];
    }
    dim3 blocks;
    dim3 threads;

    if (size <= 1024) {
        blocks.x = 1;
        threads.x = size;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }

    const float *input_data = (const float *)input->data;
    float *output_data = (float *)output->data;

    if (stream_handle) {
        minus_by_const_kernel<<<blocks, threads, 0,
                                *(cudaStream_t *)stream_handle->handle>>>(
            input_data, output_data, val, size);
    } else {
        minus_by_const_kernel<<<blocks, threads>>>(input_data, output_data, val,
                                                   size);
    }
    return 0;
}