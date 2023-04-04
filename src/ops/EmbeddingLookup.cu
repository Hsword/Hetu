#include "gpu_runtime.h"

__global__ void embedding_lookup_kernel(const float *input, const int *ids,
                                        float *output, size_t nrow,
                                        size_t length, size_t size) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size)
        return;
    int id = ids[index];
    float *output_ptr = output + length * index;
    if (id < 0 || id >= nrow) {
        for (int i = 0; i < length; ++i)
            output_ptr[i] = 0;
    } else {
        const float *input_ptr = input + length * id;
        for (int i = 0; i < length; ++i)
            output_ptr[i] = input_ptr[i];
    }
}

int DLGpuEmbeddingLookUp(const DLArrayHandle input, const DLArrayHandle ids,
                         DLArrayHandle output,
                         DLStreamHandle stream_handle = NULL) {
    assert(input->ndim == 2);
    size_t size = ArrSize(ids);
    for (int i = 0; i < output->ndim; i++) {
        if (i < output->ndim - 1) {
            assert(ids->shape[i] == output->shape[i]);
        } else {
            assert(input->shape[1] == output->shape[i]);
        }
    }
    size_t nrow = input->shape[0], length = input->shape[1];
    dim3 blocks;
    dim3 threads;
    ThreadBlock1D(threads, blocks, size);
    float *output_data = (float *)output->data;
    const float *input_data = (const float *)input->data;
    const int *id_list = (const int *)ids->data;
    if (stream_handle)
        embedding_lookup_kernel<<<blocks, threads, 0,
                                  *(cudaStream_t *)stream_handle->handle>>>(
            input_data, id_list, output_data, nrow, length, size);
    else
        embedding_lookup_kernel<<<blocks, threads>>>(
            input_data, id_list, output_data, nrow, length, size);
    return 0;
}
