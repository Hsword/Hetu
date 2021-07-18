#include "gpu_runtime.h"

__global__ void ele_mult_kernel_2(const float *matA, const float *matB,
                                  float *output, size_t size, size_t size_2) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    output[ind] = matA[ind] * matB[(int)(ind / size_2)];
}

int DLGpuDot(const DLArrayHandle matA, const DLArrayHandle matB,
             DLArrayHandle output, DLStreamHandle stream_handle = NULL) {
    size_t size_A = 1;
    assert(matA->ndim == output->ndim);
    for (index_t i = 0; i < matA->ndim; i++) {
        size_A *= matA->shape[i];
        assert(matA->shape[i] == output->shape[i]);
    }
    size_t size_B = 1;
    for (index_t i = 0; i < matB->ndim; i++) {
        size_B *= matB->shape[i];
    }
    dim3 blocks;
    dim3 threads;
    float *output_data = (float *)output->data;
    const float *matA_data = (const float *)matA->data;
    const float *matB_data = (const float *)matB->data;
    if (size_A <= 1024) {
        threads.x = size_A;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size_A + 1023) / 1024;
    }
    if (stream_handle)
        ele_mult_kernel_2<<<blocks, threads, 0,
                            *(cudaStream_t *)stream_handle->handle>>>(
            matA_data, matB_data, output_data, size_A, size_B);
    else
        ele_mult_kernel_2<<<blocks, threads>>>(matA_data, matB_data,
                                               output_data, size_A, size_B);
    return 0;
}