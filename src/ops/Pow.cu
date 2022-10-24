#include "gpu_runtime.h"

__global__ void pow_kernel(const float *input, float *output, float exp,
                           size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    output[ind] = pow(input[ind], exp);
}

int DLGpuPow(const DLArrayHandle input, DLArrayHandle output, float exp,
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
        pow_kernel<<<blocks, threads, 0,
                     *(cudaStream_t *)stream_handle->handle>>>(
            input_data, output_data, exp, size);
    else
        pow_kernel<<<blocks, threads>>>(input_data, output_data, exp, size);
    return 0;
}

__global__ void pow_grad_kernel(const float *input, const float *in_grad,
                                float *output, float exp, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    output[ind] = exp * pow(input[ind], (exp - 1)) * in_grad[ind];
}

int DLGpuPowGradient(const DLArrayHandle input, const DLArrayHandle in_grad,
                     DLArrayHandle output, float exp,
                     DLStreamHandle stream_handle = NULL) {
    size_t size = 1;
    for (index_t i = 0; i < input->ndim; i++) {
        size *= input->shape[i];
    }
    dim3 blocks;
    dim3 threads;
    const float *input_data = (const float *)input->data;
    const float *in_grad_data = (const float *)in_grad->data;
    float *output_data = (float *)output->data;
    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }
    if (stream_handle)
        pow_grad_kernel<<<blocks, threads, 0,
                          *(cudaStream_t *)stream_handle->handle>>>(
            input_data, in_grad_data, output_data, exp, size);
    else
        pow_grad_kernel<<<blocks, threads>>>(input_data, in_grad_data,
                                             output_data, exp, size);
    return 0;
}
