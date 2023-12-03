#include "gpu_runtime.h"

__global__ void ele_div_kernel(const float *matA, const float *matB,
                               float *output, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    output[ind] = matA[ind] / matB[ind];
}

int DLGpuMatrixElementwiseDivide(const DLArrayHandle matA,
                                 const DLArrayHandle matB, DLArrayHandle output,
                                 DLStreamHandle stream_handle = NULL) {
    size_t size = 1;
    for (index_t i = 0; i < matA->ndim; i++) {
        size *= matA->shape[i];
    }
    dim3 blocks;
    dim3 threads;
    float *output_data = (float *)output->data;
    const float *matA_data = (const float *)matA->data;
    const float *matB_data = (const float *)matB->data;
    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }
    if (stream_handle)
        ele_div_kernel<<<blocks, threads, 0,
                         *(cudaStream_t *)stream_handle->handle>>>(
            matA_data, matB_data, output_data, size);
    else
        ele_div_kernel<<<blocks, threads>>>(matA_data, matB_data, output_data,
                                            size);
    return 0;
}

__global__ void ele_div_handle_zero_kernel(const float *matA, const float *matB,
                                           float *output, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    if (matB[ind] == 0)
        output[ind] = matA[ind];
    else
        output[ind] = matA[ind] / matB[ind];
}

int DLGpuMatrixElementwiseDivideHandleZero(
    const DLArrayHandle matA, const DLArrayHandle matB, DLArrayHandle output,
    DLStreamHandle stream_handle = NULL) {
    size_t size = ArrSize(matA);
    float *output_data = (float *)output->data;
    const float *matA_data = (const float *)matA->data;
    const float *matB_data = (const float *)matB->data;
    dim3 blocks;
    dim3 threads;
    ThreadBlock1D(threads, blocks, size);
    if (stream_handle)
        ele_div_handle_zero_kernel<<<blocks, threads, 0,
                                     *(cudaStream_t *)stream_handle->handle>>>(
            matA_data, matB_data, output_data, size);
    else
        ele_div_handle_zero_kernel<<<blocks, threads>>>(matA_data, matB_data,
                                                        output_data, size);
    return 0;
}