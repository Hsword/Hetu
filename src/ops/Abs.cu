#include "gpu_runtime.h"

__global__ void abs_kernel(const float *input, float *output, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    output[ind] = abs(input[ind]);
}

int DLGpuAbs(const DLArrayHandle input, DLArrayHandle output,
             DLStreamHandle stream_handle = NULL) {
    size_t size = 1;
    for (index_t i = 0; i < input->ndim; i++) {
        size *= input->shape[i];
    }
    const float *input_data = (const float *)input->data;
    float *output_data = (float *)output->data;

    dim3 blocks;
    dim3 threads;
    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }
    if (stream_handle) {
        abs_kernel<<<blocks, threads, 0,
                     *(cudaStream_t *)stream_handle->handle>>>(
            input_data, output_data, size);
    } else {
        abs_kernel<<<blocks, threads>>>(input_data, output_data, size);
    }
    return 0;
}

__global__ void abs_gradient_kernel(const float *grad, const float *input,
                                    float *output, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    if (input[ind] == 0)
        output[ind] = 0;
    else
        output[ind] = grad[ind] * input[ind] / abs(input[ind]);
}

int DLGpuAbsGradient(const DLArrayHandle grad, const DLArrayHandle input,
                     DLArrayHandle output,
                     DLStreamHandle stream_handle = NULL) {
    size_t size = 1;
    for (index_t i = 0; i < grad->ndim; i++) {
        size *= grad->shape[i];
    }
    const float *grad_data = (const float *)grad->data;
    const float *input_data = (const float *)input->data;
    float *output_data = (float *)output->data;

    dim3 blocks;
    dim3 threads;
    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }
    if (stream_handle) {
        abs_gradient_kernel<<<blocks, threads, 0,
                              *(cudaStream_t *)stream_handle->handle>>>(
            grad_data, input_data, output_data, size);
    } else {
        abs_gradient_kernel<<<blocks, threads>>>(grad_data, input_data,
                                                 output_data, size);
    }
    return 0;
}
