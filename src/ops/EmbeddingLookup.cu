#include "gpu_runtime.h"

__global__ void embedding_lookup_kernel(const float *input, const int *ids,
                                        float *output, size_t size,
                                        size_t length, size_t input_row) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size)
        return;
    int id = ids[index];
    float *output_ptr = output + length * index;
    if (id < 0 || id >= input_row) {
        for (int i = 0; i < length; i++)
            output_ptr[i] = 0;
    } else {
        const float *input_ptr = input + length * id;
        for (int i = 0; i < length; i++)
            output_ptr[i] = input_ptr[i];
    }
}

int DLGpuEmbeddingLookUp(const DLArrayHandle input, const DLArrayHandle ids,
                         DLArrayHandle output,
                         DLStreamHandle stream_handle = NULL) {
    assert(input->ndim == 2);
    size_t size = 1;
    for (int i = 0; i < output->ndim; i++) {
        if (i < output->ndim - 1) {
            assert(ids->shape[i] == output->shape[i]);
        } else if (i == output->ndim - 1) {
            assert(input->shape[1] == output->shape[i]);
        }
    }
    for (int i = 0; i < ids->ndim; i++) {
        size = size * ids->shape[i];
    }
    size_t input_row = input->shape[0];
    size_t length = input->shape[1];
    dim3 blocks;
    dim3 threads;
    float *output_data = (float *)output->data;
    const float *input_data = (const float *)input->data;
    const int *id_list = (const int *)ids->data;
    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }
    if (stream_handle)
        embedding_lookup_kernel<<<blocks, threads, 0,
                                  *(cudaStream_t *)stream_handle->handle>>>(
            input_data, id_list, output_data, size, length, input_row);
    else
        embedding_lookup_kernel<<<blocks, threads>>>(
            input_data, id_list, output_data, size, length, input_row);
    return 0;
}

__global__ void array_set_zero_kernel(float *output, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    output[ind] = 0;
}
