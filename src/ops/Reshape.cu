#include "gpu_runtime.h"

extern __global__ void float_memory_copy(float *A, const float *B, size_t len);

int DLGpuReshape(const DLArrayHandle in_arr, DLArrayHandle out_arr,
                 DLStreamHandle stream_handle = NULL) {
    size_t input_size = 1;
    size_t output_size = 1;
    for (index_t i = 0; i < in_arr->ndim; i++) {
        input_size *= in_arr->shape[i];
    }
    for (index_t i = 0; i < out_arr->ndim; i++) {
        output_size *= out_arr->shape[i];
    }
    assert(input_size == output_size);
    const float *input_data = (const float *)in_arr->data;
    float *output_data = (float *)out_arr->data;
    size_t BLOCKS = (input_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (stream_handle)
        float_memory_copy<<<BLOCKS, THREADS_PER_BLOCK, 0,
                            *(cudaStream_t *)stream_handle->handle>>>(
            output_data, input_data, input_size);
    else
        float_memory_copy<<<BLOCKS, THREADS_PER_BLOCK>>>(
            output_data, input_data, input_size);
    return 0;
}