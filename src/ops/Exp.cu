#include "gpu_runtime.h"
__global__ void exp_kernel(const float *input, float *output, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    output[ind] = expf(input[ind]);
}

int DLGpuExp(const DLArrayHandle input, DLArrayHandle output,
             DLStreamHandle stream_handle = NULL) {
    size_t size = ArrSize(input);
    const float *input_data = (const float *)input->data;
    float *output_data = (float *)output->data;

    dim3 blocks;
    dim3 threads;
    ThreadBlock1D(threads, blocks, size);
    if (stream_handle) {
        exp_kernel<<<blocks, threads, 0,
                     *(cudaStream_t *)stream_handle->handle>>>(
            input_data, output_data, size);
    } else {
        exp_kernel<<<blocks, threads>>>(input_data, output_data, size);
    }
    return 0;
}
