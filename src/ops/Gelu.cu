#include "gpu_runtime.h"
#define pi 3.14159265358979323846f
#define e  2.71828182845904523536f
#define SQRT_1_2  0.70710678118654757274f  // sqrt(1/2)

__global__ void Gelu_kernel(float *input, float *output, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    output[ind] = input[ind] * 0.5f * (1.0f + erff(input[ind] * SQRT_1_2));
}

int DLGpuGelu(const DLArrayHandle input, DLArrayHandle output,
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
        Gelu_kernel<<<blocks, threads, 0,
                      *(cudaStream_t *)stream_handle->handle>>>(
            input_data, output_data, size);
    else
        Gelu_kernel<<<blocks, threads>>>(input_data, output_data, size);
    return 0;
}

__global__ void gelu_grad_kernel(const float *input, const float *in_grad,
                                 float *output, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    output[ind] = in_grad[ind]*(0.5f+0.5f*erf( input[ind]/sqrtf(2.0))+0.5f*input[ind]*(sqrtf(2.0f)*powf(e,(-0.5f*powf(input[ind],2.0f)))/sqrtf(pi)));
}

int DLGpuGeluGradient(const DLArrayHandle input, const DLArrayHandle in_grad,
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
        gelu_grad_kernel<<<blocks, threads, 0,
                           *(cudaStream_t *)stream_handle->handle>>>(
            input_data, in_grad_data, output_data, size);
    else
        gelu_grad_kernel<<<blocks, threads>>>(input_data, in_grad_data,
                                              output_data, size);
    return 0;
}
