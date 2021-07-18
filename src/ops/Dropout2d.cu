#include "gpu_runtime.h"
#include <curand.h>

__global__ void dropout2d_kernel(const float *input, float *output,
                                 const float rate, size_t size,
                                 size_t last_two) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    size_t leader = size_t(ind / last_two) * last_two;
    float keep_mask = (float)(output[leader] >= rate);
    output[ind] = input[ind] * keep_mask / (1 - rate);
}

int DLGpuDropout2d(const DLArrayHandle input, const float dropout,
                   DLArrayHandle output, unsigned long long *pseed,
                   DLStreamHandle stream_handle = NULL) {
    assert(input->ndim == 4);
    size_t size = 1;
    for (index_t i = 0; i < input->ndim; i++) {
        size *= input->shape[i];
    }
    size_t last_two_size = 1;
    last_two_size *= input->shape[input->ndim - 1];
    last_two_size *= input->shape[input->ndim - 2];

    const float *input_data = (const float *)input->data;
    float *output_data = (float *)output->data;

    curandGenerator_t gen;
    CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10));
    *pseed = time(0);
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, *pseed));
    CURAND_CALL(curandGenerateUniform(gen, output_data, size));
    CURAND_CALL(curandDestroyGenerator(gen));

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
        dropout2d_kernel<<<blocks, threads, 0,
                           *(cudaStream_t *)stream_handle->handle>>>(
            input_data, output_data, dropout, size, last_two_size);
    } else {
        dropout2d_kernel<<<blocks, threads>>>(input_data, output_data, dropout,
                                              size, last_two_size);
    }
    return 0;
}

int DLGpuDropout2dGradient(const DLArrayHandle grad, const float dropout,
                           DLArrayHandle output, unsigned long long seed,
                           DLStreamHandle stream_handle = NULL) {
    assert(grad->ndim == 4);
    size_t size = 1;
    for (index_t i = 0; i < grad->ndim; i++) {
        size *= grad->shape[i];
    }
    size_t last_two_size = 1;
    last_two_size *= grad->shape[grad->ndim - 1];
    last_two_size *= grad->shape[grad->ndim - 2];

    const float *grad_data = (const float *)grad->data;
    float *output_data = (float *)output->data;

    curandGenerator_t gen;
    CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, seed));
    CURAND_CALL(curandGenerateUniform(gen, output_data, size));
    CURAND_CALL(curandDestroyGenerator(gen));

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
        dropout2d_kernel<<<blocks, threads, 0,
                           *(cudaStream_t *)stream_handle->handle>>>(
            grad_data, output_data, dropout, size, last_two_size);
    } else {
        dropout2d_kernel<<<blocks, threads>>>(grad_data, output_data, dropout,
                                              size, last_two_size);
    }
    return 0;
}
