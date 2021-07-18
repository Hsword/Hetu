#include "gpu_runtime.h"

__global__ void leaky_relu_kernel(float *input, const float alpha,
                                  float *output, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    float s = alpha;
    if (input[ind] > 0.)
        s = 1.;
    output[ind] = input[ind] * s;
}

int DLGpuLeakyRelu(const DLArrayHandle input, const float alpha,
                   DLArrayHandle output, DLStreamHandle stream_handle = NULL) {
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
        leaky_relu_kernel<<<blocks, threads, 0,
                            *(cudaStream_t *)stream_handle->handle>>>(
            input_data, alpha, output_data, size);
    else
        leaky_relu_kernel<<<blocks, threads>>>(input_data, alpha, output_data,
                                               size);
    return 0;
}

__global__ void leaky_relu_grad_kernel(const float *input, const float *in_grad,
                                       const float alpha, float *output,
                                       size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    output[ind] = in_grad[ind];
    if (input[ind] < 0.)
        output[ind] *= alpha;
}

int DLGpuLeakyReluGradient(const DLArrayHandle input,
                           const DLArrayHandle in_grad, const float alpha,
                           DLArrayHandle output,
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
        leaky_relu_grad_kernel<<<blocks, threads, 0,
                                 *(cudaStream_t *)stream_handle->handle>>>(
            input_data, in_grad_data, alpha, output_data, size);
    else
        leaky_relu_grad_kernel<<<blocks, threads>>>(input_data, in_grad_data,
                                                    alpha, output_data, size);
    return 0;
}
