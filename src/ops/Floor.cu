#include "gpu_runtime.h"
#include <math.h>
__global__ void floor_kernel(const float *input, float *output, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    output[ind] = floor(input[ind]);
}

int DLGpuFloor(const DLArrayHandle input, DLArrayHandle output,
               DLStreamHandle stream_handle = NULL) {
    size_t size = 1;
    for (index_t i = 0; i < input->ndim; i++) {
        size *= input->shape[i];
    }
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
        floor_kernel<<<blocks, threads, 0,
                       *(cudaStream_t *)stream_handle->handle>>>(
            input_data, output_data, size);
    else
        floor_kernel<<<blocks, threads>>>(input_data, output_data, size);
    return 0;
}
