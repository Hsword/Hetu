#include "gpu_runtime.h"

__global__ void power_kernel(float *input, float *output, float p,
                             size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    output[ind] = pow(input[ind], p);
}

int DLGpuPower(const DLArrayHandle input, DLArrayHandle output, float p,
               DLStreamHandle stream_handle = NULL) {
    size_t size = ArrSize(input);
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
        power_kernel<<<blocks, threads, 0,
                       *(cudaStream_t *)stream_handle->handle>>>(
            input_data, output_data, p, size);
    else
        power_kernel<<<blocks, threads>>>(input_data, output_data, p, size);
    return 0;
}