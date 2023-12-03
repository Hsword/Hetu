#include "gpu_runtime.h"
#include "random.h"
#include <curand_kernel.h>

__global__ void dropout_kernel(const float *input, float *output,
                               HetuRandomState cudars, const float rate,
                               size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    curandStatePhilox4_32_10_t state;
    curand_init(cudars.seed, cudars.seqnum, ind, &state);
    float temp = curand_uniform(&state);
    float keep_mask = (float)(temp >= rate);
    output[ind] = input[ind] * keep_mask / (1 - rate);
}

int DLGpuDropout(const DLArrayHandle input, const float dropout,
                 DLArrayHandle output, DLStreamHandle stream_handle = NULL) {
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
    HetuRandomState &cudars = GetRandomState(1);
    if (stream_handle) {
        dropout_kernel<<<blocks, threads, 0,
                         *(cudaStream_t *)stream_handle->handle>>>(
            input_data, output_data, cudars, dropout, size);
    } else {
        dropout_kernel<<<blocks, threads>>>(input_data, output_data, cudars,
                                            dropout, size);
    }
    return 0;
}

int DLGpuDropoutGradient_recompute(const DLArrayHandle grad,
                                   const float dropout, DLArrayHandle output,
                                   unsigned long long seed_seqnum,
                                   DLStreamHandle stream_handle = NULL) {
    size_t size = 1;
    for (index_t i = 0; i < grad->ndim; i++) {
        size *= grad->shape[i];
    }
    const float *grad_data = (const float *)grad->data;
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
    HetuRandomState cudars = NewRandomState(seed_seqnum);
    if (stream_handle) {
        dropout_kernel<<<blocks, threads, 0,
                         *(cudaStream_t *)stream_handle->handle>>>(
            grad_data, output_data, cudars, dropout, size);
    } else {
        dropout_kernel<<<blocks, threads>>>(grad_data, output_data, cudars,
                                            dropout, size);
    }
    return 0;
}

__global__ void dropout_gradient_kernel(const float *grad,
                                        const float *fw_output, float *output,
                                        const float rate, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    float fw_out = fw_output[ind];
    float keep_mask = (float)(fw_out > 1e-10 || fw_out < -1e-10);
    output[ind] = grad[ind] * keep_mask / (1 - rate);
}

int DLGpuDropoutGradient(const DLArrayHandle grad,
                         const DLArrayHandle fw_output, const float dropout,
                         DLArrayHandle output,
                         DLStreamHandle stream_handle = NULL) {
    size_t size = 1;
    for (index_t i = 0; i < grad->ndim; i++) {
        size *= grad->shape[i];
    }
    const float *grad_data = (const float *)grad->data;
    const float *fw_output_data = (const float *)fw_output->data;
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
        dropout_gradient_kernel<<<blocks, threads, 0,
                                  *(cudaStream_t *)stream_handle->handle>>>(
            grad_data, fw_output_data, output_data, dropout, size);
    } else {
        dropout_gradient_kernel<<<blocks, threads>>>(
            grad_data, fw_output_data, output_data, dropout, size);
    }
    return 0;
}
