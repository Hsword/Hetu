#include "gpu_runtime.h"
#define pi 3.14159265358979323846
#define e 2.71828182845904523536

__global__ void fmod_kernel(float *input, float *output, float val,
                            size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    float input_val = input[ind];
    output[ind] = input_val - trunc(input_val / val) * val;
}

int DLGpuFmod(const DLArrayHandle input, DLArrayHandle output, float val,
              DLStreamHandle stream_handle = NULL) {
    size_t size = 1;
    for (index_t i = 0; i < input->ndim; i++) {
        size *= input->shape[i];
    }
    dim3 blocks;
    dim3 threads;
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;
    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }
    if (stream_handle)
        fmod_kernel<<<blocks, threads, 0,
                      *(cudaStream_t *)stream_handle->handle>>>(
            input_data, output_data, val, size);
    else
        fmod_kernel<<<blocks, threads>>>(input_data, output_data, val, size);
    return 0;
}
