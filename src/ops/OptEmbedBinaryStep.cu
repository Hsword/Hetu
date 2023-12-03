#include "gpu_runtime.h"

__global__ void is_positive_kernel(const float *input, float *output,
                                   size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    output[ind] = (float)(input[ind] > 0);
}

int DLGpuIsPositive(const DLArrayHandle input, DLArrayHandle output,
                    DLStreamHandle stream_handle = NULL) {
    size_t size = ArrSize(input);
    dim3 blocks;
    dim3 threads;
    ThreadBlock1D(threads, blocks, size);
    void *input_data = input->data;
    void *output_data = output->data;
    assert(stream_handle != NULL);
    cudaStream_t stream = *(cudaStream_t *)stream_handle->handle;
    is_positive_kernel<<<blocks, threads, 0, stream>>>(
        (const float *)input_data, (float *)output_data, size);
    return 0;
}

__global__ void binary_step_backward_kernel(const float *input, float *output,
                                            size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    float cur_input = abs(input[ind]);
    float res;
    if (cur_input > 1)
        res = 0;
    else if (cur_input > 0.4)
        res = 0.4;
    else
        res = 2 - 4 * cur_input;
    output[ind] = res;
}

int DLGpuBinaryStepBackward(const DLArrayHandle input, DLArrayHandle output,
                            DLStreamHandle stream_handle = NULL) {
    size_t size = ArrSize(input);
    dim3 blocks;
    dim3 threads;
    ThreadBlock1D(threads, blocks, size);
    void *input_data = input->data;
    void *output_data = output->data;
    assert(stream_handle != NULL);
    cudaStream_t stream = *(cudaStream_t *)stream_handle->handle;
    binary_step_backward_kernel<<<blocks, threads, 0, stream>>>(
        (const float *)input_data, (float *)output_data, size);
    return 0;
}