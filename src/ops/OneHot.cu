#include "gpu_runtime.h"

__global__ void onehot_kernel(const float *input, float *output,
                              size_t last_dim, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    float offset = (float)(ind % last_dim);
    float writein = 0;
    if (offset == input[ind / last_dim]) {
        writein = 1;
    } else {
        writein = 0;
    }
    output[ind] = writein;
}

int DLGpuOneHot(const DLArrayHandle input, DLArrayHandle output,
                DLStreamHandle stream_handle = NULL) {
    size_t insize = 1;
    for (int i = 0; i < input->ndim; ++i) {
        insize *= input->shape[i];
    }
    size_t last_dim = output->shape[input->ndim];
    size_t size = insize * last_dim;
    const float *input_data = (const float *)input->data;
    float *output_data = (float *)output->data;

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
        onehot_kernel<<<blocks, threads, 0,
                        *(cudaStream_t *)stream_handle->handle>>>(
            input_data, output_data, last_dim, size);
    else
        onehot_kernel<<<blocks, threads>>>(input_data, output_data, last_dim,
                                           size);
    return 0;
}