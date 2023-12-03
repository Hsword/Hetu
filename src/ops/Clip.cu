#include "gpu_runtime.h"

__global__ void clipping_kernel(float *arr, float min_value, float max_value,
                                size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    float value = arr[ind];
    value = min(value, max_value);
    value = max(value, min_value);
    arr[ind] = value;
}

int DLGpuClipping(DLArrayHandle arr, float min_value, float max_value,
                  DLStreamHandle stream_handle = NULL) {
    size_t size = ArrSize(arr);
    dim3 blocks;
    dim3 threads;
    ThreadBlock1D(threads, blocks, size);
    float *arr_data = (float *)arr->data;
    if (stream_handle)
        clipping_kernel<<<blocks, threads, 0,
                          *(cudaStream_t *)stream_handle->handle>>>(
            arr_data, min_value, max_value, size);
    else
        clipping_kernel<<<blocks, threads>>>(arr_data, min_value, max_value,
                                             size);
    return 0;
}