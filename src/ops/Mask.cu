#include "gpu_runtime.h"

__global__ void mask_kernel(const float *input_data, const int *mask_data,
                            float *output_data, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) {
        return;
    }
    if (mask_data[i])
        output_data[i] = input_data[i];
    else
        output_data[i] = 0.;
}

int DLGpuMask(const DLArrayHandle input, const DLArrayHandle mask,
              DLArrayHandle output, DLStreamHandle stream_handle) {
    size_t size = ArrSize(input);

    const float *input_data = (const float *)input->data;
    const int *mask_data = (const int *)mask->data;
    float *output_data = (float *)output->data;

    dim3 blocks;
    dim3 threads;
    ThreadBlock1D(threads, blocks, size);

    if (stream_handle) {
        mask_kernel<<<blocks, threads, 0,
                      *(cudaStream_t *)stream_handle->handle>>>(
            input_data, mask_data, output_data, size);
    } else {
        mask_kernel<<<blocks, threads>>>(input_data, mask_data, output_data,
                                         size);
    }
    return 0;
}
