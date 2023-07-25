#include "gpu_runtime.h"

__global__ void triu_tril_kernel(const float *input, float *output, bool lower,
                                 int H, int W, int diagonal, int size) {
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    int row = (ind / W) % H;
    int col = ind % W;
    bool mask = lower ? (col - row > diagonal) : (col - row < diagonal);

    output[ind] = mask ? 0 : input[ind];
}

int DLGpuTriuTril(const DLArrayHandle input, DLArrayHandle output, bool lower,
                  int diagonal, DLStreamHandle stream_handle = NULL) {
    int size = 1;
    int ndim = input->ndim;
    for (index_t i = 0; i < ndim; i++) {
        size *= input->shape[i];
    }
    int H = input->shape[ndim - 2];
    int W = input->shape[ndim - 1];

    dim3 blocks;
    dim3 threads;
    const float *input_data = (const float *)input->data;
    float *output_data = (float *)output->data;
    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }
    if (stream_handle)
        triu_tril_kernel<<<blocks, threads, 0,
                           *(cudaStream_t *)stream_handle->handle>>>(
            input_data, output_data, lower, H, W, diagonal, size);
    else
        triu_tril_kernel<<<blocks, threads>>>(input_data, output_data, lower, H,
                                              W, diagonal, size);
    return 0;
}
