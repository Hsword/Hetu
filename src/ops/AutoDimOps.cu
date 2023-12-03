#include "gpu_runtime.h"

__global__ void inplace_add_kernel(float *output, const float *adder,
                                   const float *alpha, float cons,
                                   size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    output[ind] += (adder[ind] * (*alpha) * cons);
}

int DLGpuAdd_(DLArrayHandle arr, const DLArrayHandle adder,
              const DLArrayHandle alpha, float cons,
              DLStreamHandle stream_handle) {
    size_t size = 1;
    for (index_t i = 0; i < arr->ndim; i++) {
        size *= arr->shape[i];
    }
    dim3 blocks;
    dim3 threads;
    float *output_data = (float *)arr->data;
    const float *adder_data = (const float *)adder->data;
    const float *alpha_data = (const float *)alpha->data;
    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }
    if (stream_handle)
        inplace_add_kernel<<<blocks, threads, 0,
                             *(cudaStream_t *)stream_handle->handle>>>(
            output_data, adder_data, alpha_data, cons, size);
    else
        inplace_add_kernel<<<blocks, threads>>>(output_data, adder_data,
                                                alpha_data, cons, size);
    return 0;
}

#include "gpu_runtime.h"

__global__ void inplace_div_mul_kernel(float *output, const float *alpha,
                                       float cons, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    output[ind] = output[ind] * cons / (*alpha);
}

int DLGpuDivMul(DLArrayHandle arr, const DLArrayHandle alpha, float cons,
                DLStreamHandle stream_handle) {
    size_t size = 1;
    for (index_t i = 0; i < arr->ndim; i++) {
        size *= arr->shape[i];
    }
    dim3 blocks;
    dim3 threads;
    float *output_data = (float *)arr->data;
    const float *alpha_data = (const float *)alpha->data;
    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }
    if (stream_handle)
        inplace_div_mul_kernel<<<blocks, threads, 0,
                                 *(cudaStream_t *)stream_handle->handle>>>(
            output_data, alpha_data, cons, size);
    else
        inplace_div_mul_kernel<<<blocks, threads>>>(output_data, alpha_data,
                                                    cons, size);
    return 0;
}
