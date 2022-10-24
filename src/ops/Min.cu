#include "gpu_runtime.h"

__global__ void min_kernel(float *input, float *output_val, int ROW, int COL) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= COL) {
        return;
    }
    float tmp = 1e12;
    // float tmp_idx = 0;
    for (int i = 0; i < ROW; i++) {
        if (input[col + i * COL] < tmp) {
            tmp = input[col + i * COL];
            // tmp_idx=i;
        }
    }
    // output_idx[col] = tmp_idx;
    output_val[col] = tmp;
}

int DLGpuMin(const DLArrayHandle input, DLArrayHandle output_val, int dim,
             DLStreamHandle stream_handle = NULL) {
    assert(dim == 0); // only support dim=0 now
    float *input_data = (float *)input->data;
    // float* output_idx_data=(float*)output_idx->data;
    float *output_val_data = (float *)output_val->data;

    dim3 blocks;
    dim3 threads;

    int ROW = input->shape[0];
    int COL = input->shape[1];

    if (COL <= 1024) {
        blocks.x = 1;
        threads.x = COL;
    } else {
        blocks.x = (COL + 1023) / 1024;
        threads.x = 1024;
    }

    if (stream_handle) {
        min_kernel<<<blocks, threads, 0,
                     *(cudaStream_t *)stream_handle->handle>>>(
            input_data, output_val_data, ROW, COL);
    } else {
        min_kernel<<<blocks, threads>>>(input_data, output_val_data, ROW, COL);
    }

    return 0;
}

__global__ void min_mat_kernel(float *matA, float *matB, float *output,
                               size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    output[ind] = matA[ind];
    if (matA[ind] > matB[ind])
        output[ind] = matB[ind];
}

int DLGpuMinMat(const DLArrayHandle matA, const DLArrayHandle matB,
                DLArrayHandle output, DLStreamHandle stream_handle = NULL) {
    float *matA_data = (float *)matA->data;
    float *matB_data = (float *)matB->data;
    float *output_data = (float *)output->data;

    size_t size = 1;
    for (index_t i = 0; i < matA->ndim; i++) {
        size *= matA->shape[i];
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
        min_mat_kernel<<<blocks, threads, 0,
                         *(cudaStream_t *)stream_handle->handle>>>(
            matA_data, matB_data, output_data, size);
    else
        min_mat_kernel<<<blocks, threads>>>(matA_data, matB_data, output_data,
                                            size);

    return 0;
}
