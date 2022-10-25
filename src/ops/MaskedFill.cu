#include "gpu_runtime.h"

__global__ void masked_fill_kernel(const float *input, const float *mask,
                                   float val, float *output, int size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind > size) {
        return;
    }
    if (mask[ind] == 1)
        output[ind] = val;
    else
        output[ind] = input[ind];
}

int DLGpuMaskedFill(const DLArrayHandle input, const DLArrayHandle mask,
                    float val, DLArrayHandle output,
                    DLStreamHandle stream_handle = NULL) {
    assert(input->ndim == output->ndim);
    int size = 1;
    for (int i = 0; i < input->ndim; i++) {
        size *= input->shape[i];
    }

    const float *input_data = (const float *)input->data;
    const float *mask_data = (const float *)mask->data;
    float *output_data = (float *)output->data;

    dim3 blocks;
    dim3 threads;

    if (size <= 1024) {
        blocks.x = 1;
        threads.x = size;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }

    if (stream_handle) {
        masked_fill_kernel<<<blocks, threads, 0,
                             *(cudaStream_t *)stream_handle->handle>>>(
            input_data, mask_data, val, output_data, size);
    } else {
        masked_fill_kernel<<<blocks, threads>>>(input_data, mask_data, val,
                                                output_data, size);
    }
    return 0;
}
