#include "gpu_runtime.h"

__global__ void where_kernel(const float *cond, const float *arr1,
                             const float *arr2, float *output, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    output[ind] = cond[ind] ? arr1[ind] : arr2[ind];
}

int DLGpuWhere(const DLArrayHandle cond, const DLArrayHandle arr1,
               const DLArrayHandle arr2, DLArrayHandle output,
               DLStreamHandle stream_handle = NULL) {
    size_t size = 1;
    for (index_t i = 0; i < cond->ndim; i++) {
        size *= cond->shape[i];
    }
    dim3 blocks;
    dim3 threads;
    const float *cond_data = (const float *)cond->data;
    const float *arr1_data = (const float *)arr1->data;
    const float *arr2_data = (const float *)arr2->data;
    float *output_data = (float *)output->data;
    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }
    if (stream_handle)
        where_kernel<<<blocks, threads, 0,
                       *(cudaStream_t *)stream_handle->handle>>>(
            cond_data, arr1_data, arr2_data, output_data, size);
    else
        where_kernel<<<blocks, threads>>>(cond_data, arr1_data, arr2_data,
                                          output_data, size);
    return 0;
}

__global__ void where_const_kernel(const float *cond, const float *arr1,
                                   float const_attr, float *output,
                                   size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    output[ind] = cond[ind] ? arr1[ind] : const_attr;
}

int DLGpuWhereConst(const DLArrayHandle cond, const DLArrayHandle arr1,
                    float const_attr, DLArrayHandle output,
                    DLStreamHandle stream_handle = NULL) {
    size_t size = 1;
    for (index_t i = 0; i < cond->ndim; i++) {
        size *= cond->shape[i];
    }
    dim3 blocks;
    dim3 threads;
    const float *cond_data = (const float *)cond->data;
    const float *arr1_data = (const float *)arr1->data;
    float *output_data = (float *)output->data;
    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }
    if (stream_handle)
        where_const_kernel<<<blocks, threads, 0,
                             *(cudaStream_t *)stream_handle->handle>>>(
            cond_data, arr1_data, const_attr, output_data, size);
    else
        where_const_kernel<<<blocks, threads>>>(cond_data, arr1_data,
                                                const_attr, output_data, size);
    return 0;
}