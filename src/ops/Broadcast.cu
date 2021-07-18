#include "gpu_runtime.h"

__global__ void broadcast_to_kernel(const float *input_data, float *output_data,
                                    size_t input_size, size_t output_size) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= output_size)
        return;
    output_data[id] = input_data[id % input_size];
}

int DLGpuBroadcastTo(const DLArrayHandle input, DLArrayHandle output,
                     DLStreamHandle stream_handle = NULL) {
    for (index_t i = 0; i < input->ndim; i++) {
        assert((input->shape[i]) == (output->shape[i + 1]));
    }
    size_t input_size = 1;
    for (index_t i = 0; i < input->ndim; i++) {
        input_size *= input->shape[i];
    }
    size_t output_size = 1;
    for (index_t i = 0; i < output->ndim; i++) {
        output_size *= output->shape[i];
    }
    size_t BLOCKS = (output_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (stream_handle) {
        cudaStream_t *s = (cudaStream_t *)(stream_handle->handle);
        broadcast_to_kernel<<<BLOCKS, THREADS_PER_BLOCK, 0, *s>>>(
            (const float *)(input->data), (float *)(output->data), input_size,
            output_size);
    } else
        broadcast_to_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(
            (const float *)(input->data), (float *)(output->data), input_size,
            output_size);
    return 0;
}
