#include "gpu_runtime.h"

__global__ void softmax_kernel(int nrow, int ncol, const float *input,
                               float *output) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    for (int thread_id = id; thread_id < nrow;
         thread_id += blockDim.x * gridDim.x) {
        float maxval = input[thread_id * ncol];
        // Find max for a row.
        for (int x = 1; x < ncol; ++x) {
            maxval = max(maxval, input[thread_id * ncol + x]);
        }
        // Deduct by max for a row, and raise to exp.
        float sum = 0;
        for (int x = 0; x < ncol; ++x) {
            sum += exp(input[thread_id * ncol + x] - maxval);
        }
        for (int x = 0; x < ncol; ++x) {
            output[thread_id * ncol + x] =
                exp(input[thread_id * ncol + x] - maxval) / sum;
        }
    }
}

int DLGpuSoftmax(const DLArrayHandle input, DLArrayHandle output,
                 DLStreamHandle stream_handle = NULL) {
    assert(input->ndim == output->ndim);
    int nrow = 1;
    size_t ndim = input->ndim;
    for (int i = 0; i < ndim - 1; ++i) {
        assert(input->shape[i] == output->shape[i]);
        nrow *= input->shape[i];
    }
    assert(input->shape[ndim - 1] == output->shape[ndim - 1]);
    int ncol = input->shape[ndim - 1];
    const float *input_data = (const float *)input->data;
    float *output_data = (float *)output->data;
    if (stream_handle)
        softmax_kernel<<<1, THREADS_PER_BLOCK, 0,
                         *(cudaStream_t *)stream_handle->handle>>>(
            nrow, ncol, input_data, output_data);
    else
        softmax_kernel<<<1, THREADS_PER_BLOCK>>>(nrow, ncol, input_data,
                                                 output_data);
    return 0;
}
