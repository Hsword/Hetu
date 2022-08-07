#include "gpu_runtime.h"

__global__ void less_const_kernel(float *input, float *output, float threshold,
                                  size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    output[ind] = (input[ind] < threshold);
}

int DLGpuNumLessThan(const DLArrayHandle input, DLArrayHandle middle,
                     DLArrayHandle output, float threshold, int *axes,
                     int num_ax, DLStreamHandle stream_handle = NULL) {
    size_t size = ArrSize(input);
    float *input_data = (float *)input->data;
    float *middle_data = (float *)middle->data;
    dim3 blocks;
    dim3 threads;
    ThreadBlock1D(threads, blocks, size);
    if (stream_handle)
        less_const_kernel<<<blocks, threads, 0,
                            *(cudaStream_t *)stream_handle->handle>>>(
            input_data, middle_data, threshold, size);
    else
        less_const_kernel<<<blocks, threads>>>(input_data, middle_data,
                                               threshold, size);
    return DLGpuReduceSum(middle, output, axes, num_ax, stream_handle);
}

__global__ void set_less_const_kernel(float *arr, float threshold,
                                      size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    if (abs(arr[ind]) < threshold) {
        arr[ind] = 0;
    }
}

int DLGpuSetLessThan(const DLArrayHandle arr, float threshold,
                     DLStreamHandle stream_handle = NULL) {
    size_t size = ArrSize(arr);
    float *arr_data = (float *)arr->data;
    dim3 blocks;
    dim3 threads;
    ThreadBlock1D(threads, blocks, size);
    if (stream_handle)
        set_less_const_kernel<<<blocks, threads, 0,
                                *(cudaStream_t *)stream_handle->handle>>>(
            arr_data, threshold, size);
    else
        set_less_const_kernel<<<blocks, threads>>>(arr_data, threshold, size);
    return 0;
}
