#include "gpu_runtime.h"

__global__ void const_pow_kernel(const float *input, float *output, float val,
                                 size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    output[ind] = pow(val, input[ind]);
}

int DLGpuConstPow(const DLArrayHandle input, float val, DLArrayHandle output,
                  DLStreamHandle stream_handle = NULL) {
    size_t size = 1;
    for (index_t i = 0; i < input->ndim; i++) {
        size *= input->shape[i];
    }
    dim3 blocks;
    dim3 threads;
    const float *input_data = (const float *)input->data;
    float *output_data = (float *)output->data;
    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }
    if (stream_handle)
        const_pow_kernel<<<blocks, threads, 0,
                           *(cudaStream_t *)stream_handle->handle>>>(
            input_data, output_data, val, size);
    else
        const_pow_kernel<<<blocks, threads>>>(input_data, output_data, val,
                                              size);
    return 0;
}

__global__ void const_pow_gradient_kernel(const float *input, const float *grad,
                                          float *output, float val,
                                          size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    output[ind] = input[ind] * grad[ind] * log(val);
}

int DLGpuConstPowGradient(const DLArrayHandle input, const DLArrayHandle grad,
                          float val, DLArrayHandle output,
                          DLStreamHandle stream_handle = NULL) {
    size_t size = 1;
    for (index_t i = 0; i < input->ndim; i++) {
        size *= input->shape[i];
    }
    dim3 blocks;
    dim3 threads;
    const float *input_data = (const float *)input->data;
    const float *grad_data = (const float *)grad->data;
    float *output_data = (float *)output->data;
    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }
    if (stream_handle)
        const_pow_gradient_kernel<<<blocks, threads, 0,
                                    *(cudaStream_t *)stream_handle->handle>>>(
            input_data, grad_data, output_data, val, size);
    else
        const_pow_gradient_kernel<<<blocks, threads>>>(input_data, grad_data,
                                                       output_data, val, size);
    return 0;
}