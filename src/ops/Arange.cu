#include "gpu_runtime.h"

__global__ void range_kernel(float start, float step, int64_t size,
                             float *out) {
    CUDA_KERNEL_LOOP(index, size) {
        out[index] = start + step * index;
    }
}

int DLGpuArange(float start, float end, float step, DLArrayHandle output,
                DLStreamHandle stream_handle = NULL) {
    int size = output->shape[0];
    dim3 blocks;
    dim3 threads;
    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }

    if (stream_handle) {
        cudaStream_t cu_stream = (*(cudaStream_t *)(stream_handle->handle));
        range_kernel<<<blocks, threads, 0, cu_stream>>>(start, step, size,
                                                        (float *)output->data);
    } else {
        range_kernel<<<blocks, threads>>>(start, step, size,
                                          (float *)output->data);
    }

    return 0;
}