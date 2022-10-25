#include "gpu_runtime.h"

__global__ void clamp_kernel(const float *input, float min, float max,
                             float *output, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    if (min > max) {
        output[ind] = max;
        return;
    }
    if (input[ind] < min)
        output[ind] = min;
    else if (input[ind] > max)
        output[ind] = max;
    else
        output[ind] = input[ind];
}

__global__ void clamp_min_kernel(const float *input, float min, float *output,
                                 size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    if (input[ind] < min)
        output[ind] = min;
    else
        output[ind] = input[ind];
}

__global__ void clamp_max_kernel(const float *input, float max, float *output,
                                 size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    if (input[ind] > max)
        output[ind] = max;
    else
        output[ind] = input[ind];
}

__global__ void clamp_mat_kernel(const float *input, const float *min,
                                 const float *max, float *output, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    if (min && input[ind] < min[ind])
        output[ind] = min[ind];
    else if (max && input[ind] > max[ind])
        output[ind] = max[ind];
    else
        output[ind] = input[ind];
}

int DLGpuClamp(const DLArrayHandle input, float min, float max,
               DLArrayHandle output, DLStreamHandle stream_handle = NULL) {
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;
    int size = 1;
    for (int i = 0; i < input->ndim; i++) {
        size *= input->shape[i];
    }

    dim3 blocks;
    dim3 threads;

    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }
    if (stream_handle)
        clamp_kernel<<<blocks, threads, 0,
                       *(cudaStream_t *)stream_handle->handle>>>(
            input_data, min, max, output_data, size);
    else
        clamp_kernel<<<blocks, threads>>>(input_data, min, max, output_data,
                                          size);
    return 0;
}

int DLGpuClampMin(const DLArrayHandle input, float min, DLArrayHandle output,
                  DLStreamHandle stream_handle = NULL) {
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;
    int size = 1;
    for (int i = 0; i < input->ndim; i++) {
        size *= input->shape[i];
    }

    dim3 blocks;
    dim3 threads;

    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }
    if (stream_handle)
        clamp_min_kernel<<<blocks, threads, 0,
                           *(cudaStream_t *)stream_handle->handle>>>(
            input_data, min, output_data, size);
    else
        clamp_min_kernel<<<blocks, threads>>>(input_data, min, output_data,
                                              size);
    return 0;
}

int DLGpuClampMax(const DLArrayHandle input, float max, DLArrayHandle output,
                  DLStreamHandle stream_handle = NULL) {
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;
    int size = 1;
    for (int i = 0; i < input->ndim; i++) {
        size *= input->shape[i];
    }

    dim3 blocks;
    dim3 threads;

    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }
    if (stream_handle)
        clamp_max_kernel<<<blocks, threads, 0,
                           *(cudaStream_t *)stream_handle->handle>>>(
            input_data, max, output_data, size);
    else
        clamp_max_kernel<<<blocks, threads>>>(input_data, max, output_data,
                                              size);
    return 0;
}

int DLGpuClampMat(const DLArrayHandle input, const DLArrayHandle min_mat,
                  const DLArrayHandle max_mat, DLArrayHandle output,
                  DLStreamHandle stream_handle = NULL) {
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;
    float *min_data = NULL;
    float *max_data = NULL;
    if (min_mat)
        min_data = (float *)min_mat->data;
    if (max_mat)
        max_data = (float *)max_mat->data;
    int size = 1;
    for (int i = 0; i < input->ndim; i++) {
        size *= input->shape[i];
    }

    dim3 blocks;
    dim3 threads;

    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }
    if (stream_handle)
        clamp_mat_kernel<<<blocks, threads, 0,
                           *(cudaStream_t *)stream_handle->handle>>>(
            input_data, min_data, max_data, output_data, size);
    else
        clamp_mat_kernel<<<blocks, threads>>>(input_data, min_data, max_data,
                                              output_data, size);
    return 0;
}
