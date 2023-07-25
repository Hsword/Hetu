#include "gpu_runtime.h"
#include <math.h>
__global__ void eye_kernel(float *output, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    size_t nrow = ind / size;
    size_t ncol = ind % size;
    if (nrow == ncol)
        output[ind] = 1;
    else
        output[ind] = 0;
}

int DLGpuEye(DLArrayHandle output, DLStreamHandle stream_handle = NULL) {
    size_t size = output->shape[0] * output->shape[1];
    dim3 blocks;
    dim3 threads;
    float *output_data = (float *)output->data;
    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }
    if (stream_handle)
        eye_kernel<<<blocks, threads, 0,
                     *(cudaStream_t *)stream_handle->handle>>>(output_data,
                                                               size);
    else
        eye_kernel<<<blocks, threads>>>(output_data, size);
    return 0;
}
