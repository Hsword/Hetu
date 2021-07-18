#include "gpu_runtime.h"

__global__ void conv2d_broadcast_to_kernel(size_t nthreads,
                                           const float *input_data,
                                           float *output_data,
                                           size_t input_size,
                                           size_t output_size) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= nthreads)
        return;
    size_t input_id = (id % (input_size * output_size)) / output_size;
    output_data[id] = input_data[input_id];
}

int DLGpuConv2d_broadcast_to(const DLArrayHandle input_x,
                             DLArrayHandle output_y,
                             DLStreamHandle stream_handle = NULL) {
    assert(input_x->shape[0] == output_y->shape[1]);
    const float *input_data = (const float *)input_x->data;
    float *output_data = (float *)output_y->data;
    size_t batch_size = output_y->shape[0];
    size_t input_size = input_x->shape[0];
    size_t output_size = (output_y->shape[2]) * (output_y->shape[3]);
    size_t nthreads = batch_size * input_size * output_size;
    size_t BLOCKS = (nthreads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (stream_handle)
        conv2d_broadcast_to_kernel<<<BLOCKS, THREADS_PER_BLOCK, 0,
                                     *(cudaStream_t *)stream_handle->handle>>>(
            nthreads, input_data, output_data, input_size, output_size);
    else
        conv2d_broadcast_to_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(
            nthreads, input_data, output_data, input_size, output_size);
    return 0;
}