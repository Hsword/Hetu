#include "gpu_runtime.h"

__global__ void sigmoid_kernel(float *input, float *output, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    output[ind] = 1.0f / (1.0f + exp(-input[ind]));
}

int DLGpuSigmoid(const DLArrayHandle input, DLArrayHandle output,
                 DLStreamHandle stream_handle = NULL) {
    size_t size = 1;
    for (index_t i = 0; i < input->ndim; i++) {
        size *= input->shape[i];
    }
    dim3 blocks;
    dim3 threads;
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;
    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }
    if (stream_handle)
        sigmoid_kernel<<<blocks, threads, 0,
                         *(cudaStream_t *)stream_handle->handle>>>(
            input_data, output_data, size);
    else
        sigmoid_kernel<<<blocks, threads>>>(input_data, output_data, size);
    return 0;
}