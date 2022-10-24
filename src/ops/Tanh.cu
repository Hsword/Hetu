#include "gpu_runtime.h"

__global__ void tanh_kernel(const float *input, float *output, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    output[ind] = tanhf(input[ind]);
}

int DLGpuTanh(const DLArrayHandle input, DLArrayHandle output,
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
        tanh_kernel<<<blocks, threads, 0,
                      *(cudaStream_t *)stream_handle->handle>>>(
            input_data, output_data, size);
    else
        tanh_kernel<<<blocks, threads>>>(input_data, output_data, size);
    return 0;
}

__global__ void tanh_gradient_kernel(const float *forward, const float *grad,
                                     float *output, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    output[ind] = (1 - forward[ind] * forward[ind]) * grad[ind];
}

int DLGpuTanhGradient(const DLArrayHandle forward, const DLArrayHandle grad,
                      DLArrayHandle output,
                      DLStreamHandle stream_handle = NULL) {
    size_t size = 1;
    for (index_t i = 0; i < forward->ndim; i++) {
        size *= forward->shape[i];
    }
    dim3 blocks;
    dim3 threads;
    const float *forward_data = (const float *)forward->data;
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
        tanh_gradient_kernel<<<blocks, threads, 0,
                               *(cudaStream_t *)stream_handle->handle>>>(
            forward_data, grad_data, output_data, size);
    else
        tanh_gradient_kernel<<<blocks, threads>>>(forward_data, grad_data,
                                                  output_data, size);
    return 0;
}