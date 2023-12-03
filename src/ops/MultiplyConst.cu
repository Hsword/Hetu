#include "gpu_runtime.h"

__global__ void mult_const_kernel(const float *input, float *output,
                                  float value, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    output[ind] = input[ind] * value;
}

int DLGpuMatrixMultiplyByConst(const DLArrayHandle input, float val,
                               DLArrayHandle output,
                               DLStreamHandle stream_handle = NULL) {
    int dev_id = (input->ctx).device_id;
    cudaSetDevice(dev_id);
    size_t size = 1;
    for (index_t i = 0; i < input->ndim; i++) {
        size *= input->shape[i];
    }
    dim3 blocks;
    dim3 threads;
    float *output_data = (float *)output->data;
    const float *input_data = (const float *)input->data;
    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }
    if (stream_handle)
        mult_const_kernel<<<blocks, threads, 0,
                            *(cudaStream_t *)stream_handle->handle>>>(
            input_data, output_data, val, size);
    else
        mult_const_kernel<<<blocks, threads>>>(input_data, output_data, val,
                                               size);
    return 0;
}

__global__ void mult_const_int_kernel(const int *input, int *output, int value,
                                      size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    output[ind] = input[ind] * value;
}

int DLGpuMatrixMultiplyByConstInt(const DLArrayHandle input, int val,
                                  DLArrayHandle output,
                                  DLStreamHandle stream_handle = NULL) {
    int dev_id = (input->ctx).device_id;
    cudaSetDevice(dev_id);
    size_t size = 1;
    for (index_t i = 0; i < input->ndim; i++) {
        size *= input->shape[i];
    }
    dim3 blocks;
    dim3 threads;
    ThreadBlock1D(threads, blocks, size);
    int *output_data = (int *)output->data;
    const int *input_data = (const int *)input->data;
    if (stream_handle)
        mult_const_int_kernel<<<blocks, threads, 0,
                                *(cudaStream_t *)stream_handle->handle>>>(
            input_data, output_data, val, size);
    else
        mult_const_int_kernel<<<blocks, threads>>>(input_data, output_data, val,
                                                   size);
    return 0;
}
