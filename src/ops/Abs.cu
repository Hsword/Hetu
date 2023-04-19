#include "gpu_runtime.h"

__global__ void abs_kernel(const float *input, float *output, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    float result;
    if (input[ind] >= 0)
        result = input[ind];
    else
        result = -input[ind];
    output[ind] = result;
}

int DLGpuAbs(const DLArrayHandle input, DLArrayHandle output,
             DLStreamHandle stream_handle = NULL) {
    size_t size = ArrSize(input);
    dim3 blocks;
    dim3 threads;
    ThreadBlock1D(threads, blocks, size);
    const float *input_data = (const float *)input->data;
    float *output_data = (float *)output->data;
    if (stream_handle)
        abs_kernel<<<blocks, threads, 0,
                     *(cudaStream_t *)stream_handle->handle>>>(
            input_data, output_data, size);
    else
        abs_kernel<<<blocks, threads>>>(input_data, output_data, size);
    return 0;
}
