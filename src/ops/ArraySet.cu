#include "gpu_runtime.h"

__global__ void array_set_kernel(float *output, float value, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    output[ind] = value;
}

int DLGpuArraySet(DLArrayHandle arr, float value,
                  DLStreamHandle stream_handle) {
    size_t size = 1;
    for (index_t i = 0; i < arr->ndim; i++) {
        size *= arr->shape[i];
    }
    dim3 blocks;
    dim3 threads;
    float *output_data = (float *)arr->data;
    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }
    if (stream_handle)
        array_set_kernel<<<blocks, threads, 0,
                           *(cudaStream_t *)stream_handle->handle>>>(
            output_data, value, size);
    else
        array_set_kernel<<<blocks, threads>>>(output_data, value, size);
    return 0;
}
