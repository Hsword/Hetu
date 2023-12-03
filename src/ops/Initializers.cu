#include "gpu_runtime.h"
#include "random.h"
#include <curand_kernel.h>

__global__ void init_normal_kernel(float *arr, const float mean,
                                   const float stddev, HetuRandomState cudars,
                                   size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    curandStatePhilox4_32_10_t state;
    curand_init(cudars.seed, cudars.seqnum, ind, &state);
    arr[ind] = curand_normal(&state) * stddev + mean;
}

int DLGpuNormalInit(DLArrayHandle arr, const float mean, const float stddev,
                    DLStreamHandle stream_handle = NULL) {
    size_t size = ArrSize(arr);
    float *arr_data = (float *)arr->data;

    dim3 blocks;
    dim3 threads;
    ThreadBlock1D(threads, blocks, size);
    HetuRandomState &cudars = GetRandomState(1);
    if (stream_handle) {
        init_normal_kernel<<<blocks, threads, 0,
                             *(cudaStream_t *)stream_handle->handle>>>(
            arr_data, mean, stddev, cudars, size);
    } else {
        init_normal_kernel<<<blocks, threads>>>(arr_data, mean, stddev, cudars,
                                                size);
    }

    return 0;
}

__global__ void init_scale_kernel(float *arr, const float lb, const float ub,
                                  HetuRandomState cudars, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    curandStatePhilox4_32_10_t state;
    curand_init(cudars.seed, cudars.seqnum, ind, &state);
    arr[ind] = curand_uniform(&state) * (ub - lb) + lb;
}

int DLGpuUniformInit(DLArrayHandle arr, const float lb, const float ub,
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
    HetuRandomState &cudars = GetRandomState(1);
    if (stream_handle) {
        init_scale_kernel<<<blocks, threads, 0,
                            *(cudaStream_t *)stream_handle->handle>>>(
            arr_data, lb, ub, cudars, size);
    } else {
        init_scale_kernel<<<blocks, threads>>>(arr_data, lb, ub, cudars, size);
    }

    return 0;
}

__global__ void truncated_normal_kernel(float *arr, const float mean,
                                        const float stddev,
                                        HetuRandomState cudars, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    curandStatePhilox4_32_10_t state;
    bool not_found = true;
    // here we use different sequences instead of offsets
    // to avoid using the same random number
    curand_init(cudars.seed, cudars.seqnum + ind, 0, &state);
    float temp;
    while (not_found) {
        temp = curand_normal(&state);
        not_found = (temp < -2 || temp > 2);
    }
    arr[ind] = temp * stddev + mean;
}

int DLGpuTruncatedNormalInit(DLArrayHandle arr, const float mean,
                             const float stddev,
                             DLStreamHandle stream_handle = NULL) {
    size_t size = ArrSize(arr);
    float *arr_data = (float *)arr->data;

    dim3 blocks;
    dim3 threads;
    ThreadBlock1D(threads, blocks, size);
    HetuRandomState &cudars = GetRandomState(size);
    if (stream_handle) {
        truncated_normal_kernel<<<blocks, threads, 0,
                                  *(cudaStream_t *)stream_handle->handle>>>(
            arr_data, mean, stddev, cudars, size);
    } else {
        truncated_normal_kernel<<<blocks, threads>>>(arr_data, mean, stddev,
                                                     cudars, size);
    }

    return 0;
}

__global__ void reversed_truncated_normal_kernel(float *arr, const float mean,
                                                 const float stddev,
                                                 HetuRandomState cudars,
                                                 size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    curandStatePhilox4_32_10_t state;
    bool not_found = true;
    // here we use different sequences instead of offsets
    // to avoid using the same random number
    curand_init(cudars.seed, cudars.seqnum + ind, 0, &state);
    float temp;
    while (not_found) {
        temp = curand_normal(&state);
        not_found = (temp > -2 && temp < 2);
    }
    arr[ind] = temp * stddev + mean;
}

int DLGpuReversedTruncatedNormalInit(DLArrayHandle arr, const float mean,
                                     const float stddev,
                                     DLStreamHandle stream_handle = NULL) {
    size_t size = ArrSize(arr);
    float *arr_data = (float *)arr->data;

    dim3 blocks;
    dim3 threads;
    ThreadBlock1D(threads, blocks, size);
    HetuRandomState &cudars = GetRandomState(size);
    if (stream_handle) {
        reversed_truncated_normal_kernel<<<
            blocks, threads, 0, *(cudaStream_t *)stream_handle->handle>>>(
            arr_data, mean, stddev, cudars, size);
    } else {
        reversed_truncated_normal_kernel<<<blocks, threads>>>(
            arr_data, mean, stddev, cudars, size);
    }

    return 0;
}

__global__ void gumbel_sample_kernel(float *arr, HetuRandomState cudars,
                                     size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    curandStatePhilox4_32_10_t state;
    curand_init(cudars.seed, cudars.seqnum, ind, &state);
    float value = curand_uniform(&state);
    float epsilon = 1e-12;
    value = max(value, epsilon);
    value = -log(value);
    value = max(value, epsilon);
    arr[ind] = -log(value);
}

int DLGpuGumbelInit(DLArrayHandle arr, DLStreamHandle stream_handle = NULL) {
    size_t size = ArrSize(arr);
    float *arr_data = (float *)arr->data;

    dim3 blocks;
    dim3 threads;
    ThreadBlock1D(threads, blocks, size);
    HetuRandomState &cudars = GetRandomState(1);
    if (stream_handle) {
        gumbel_sample_kernel<<<blocks, threads, 0,
                               *(cudaStream_t *)stream_handle->handle>>>(
            arr_data, cudars, size);
    } else {
        gumbel_sample_kernel<<<blocks, threads>>>(arr_data, cudars, size);
    }

    return 0;
}

__global__ void random_int_kernel(int *arr, const int lb, const int ub,
                                  HetuRandomState cudars, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    curandStatePhilox4_32_10_t state;
    curand_init(cudars.seed, cudars.seqnum, ind, &state);
    float temp = curand_uniform(&state) * (ub - lb) + lb;
    int result = (int)temp;
    result = min(result, ub - 1);
    result = max(result, lb);
    arr[ind] = result;
}

int DLGpuRandomInt(DLArrayHandle arr, const int lb, const int ub,
                   DLStreamHandle stream_handle = NULL) {
    size_t size = ArrSize(arr);
    int *arr_data = (int *)arr->data;

    dim3 blocks;
    dim3 threads;
    ThreadBlock1D(threads, blocks, size);
    HetuRandomState &cudars = GetRandomState(1);
    if (stream_handle) {
        random_int_kernel<<<blocks, threads, 0,
                            *(cudaStream_t *)stream_handle->handle>>>(
            arr_data, lb, ub, cudars, size);
    } else {
        random_int_kernel<<<blocks, threads>>>(arr_data, lb, ub, cudars, size);
    }

    return 0;
}
