#include "gpu_runtime.h"
#include <curand.h>
#include <curand_kernel.h>

int DLGpuNormalInit(DLArrayHandle arr, const float mean, const float stddev,
                    unsigned long long seed,
                    DLStreamHandle stream_handle = NULL) {
    size_t size = 1;
    for (index_t i = 0; i < arr->ndim; i++) {
        size *= arr->shape[i];
    }
    float *arr_data = (float *)arr->data;

    curandGenerator_t gen;
    CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, seed));
    if (stream_handle)
        CURAND_CALL(
            curandSetStream(gen, *(cudaStream_t *)stream_handle->handle));
    CURAND_CALL(curandGenerateNormal(gen, arr_data, size, mean, stddev));
    CURAND_CALL(curandDestroyGenerator(gen));

    return 0;
}

__global__ void init_scale_kernel(float *arr, const float lb, const float ub,
                                  unsigned long long seed, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    curandStatePhilox4_32_10_t state;
    curand_init(seed + ind, 0, 0, &state);
    arr[ind] = curand_uniform(&state) * (ub - lb) + lb;
}

int DLGpuUniformInit(DLArrayHandle arr, const float lb, const float ub,
                     unsigned long long seed,
                     DLStreamHandle stream_handle = NULL) {
    size_t size = 1;
    for (index_t i = 0; i < arr->ndim; i++) {
        size *= arr->shape[i];
    }
    float *arr_data = (float *)arr->data;

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
        init_scale_kernel<<<blocks, threads, 0,
                            *(cudaStream_t *)stream_handle->handle>>>(
            arr_data, lb, ub, seed, size);
    } else {
        init_scale_kernel<<<blocks, threads>>>(arr_data, lb, ub, seed, size);
    }

    return 0;
}

__global__ void truncated_normal_kernel(float *arr, const float mean,
                                        const float stddev,
                                        unsigned long long seed, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    curandStatePhilox4_32_10_t state;
    bool not_found = true;
    curand_init(seed + ind, 0, 0, &state);
    float temp;
    while (not_found) {
        temp = curand_normal(&state);
        not_found = (temp < -2 || temp > 2);
    }
    arr[ind] = temp * stddev + mean;
}

int DLGpuTruncatedNormalInit(DLArrayHandle arr, const float mean,
                             const float stddev, unsigned long long seed,
                             DLStreamHandle stream_handle = NULL) {
    size_t size = 1;
    for (index_t i = 0; i < arr->ndim; i++) {
        size *= arr->shape[i];
    }
    float *arr_data = (float *)arr->data;

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
        truncated_normal_kernel<<<blocks, threads, 0,
                                  *(cudaStream_t *)stream_handle->handle>>>(
            arr_data, mean, stddev, seed, size);
    } else {
        truncated_normal_kernel<<<blocks, threads>>>(arr_data, mean, stddev,
                                                     seed, size);
    }

    return 0;
}
