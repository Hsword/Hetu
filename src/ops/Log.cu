#include "gpu_runtime.h"

__global__ void log_kernel(const float *input_data, float *output_data,
                           float eps, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) {
        return;
    }
    output_data[i] = log(max(input_data[i], eps));
}

__global__ void log_grad_kernel(const float *output_grad_data,
                                float *input_data, float *input_grad_data,
                                float eps, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) {
        return;
    }
    float tmp = output_grad_data[i] / max(input_data[i], eps);
    input_grad_data[i] = tmp;
}

int DLGpuLog(const DLArrayHandle input, DLArrayHandle output, float eps,
             DLStreamHandle stream_handle) {
    assert(input->ndim == output->ndim);
    int size = 1;
    for (int i = 0; i < input->ndim; i++) {
        size *= input->shape[i];
    }

    const float *input_data = (const float *)input->data;
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
        log_kernel<<<blocks, threads, 0,
                     *(cudaStream_t *)stream_handle->handle>>>(
            input_data, output_data, eps, size);
    } else {
        log_kernel<<<blocks, threads>>>(input_data, output_data, eps, size);
    }
    return 0;
}

int DLGpuLogGrad(const DLArrayHandle output_grad, DLArrayHandle input,
                 DLArrayHandle input_grad, float eps,
                 DLStreamHandle stream_handle) {
    assert(output_grad->ndim == input->ndim);
    assert(output_grad->ndim == input_grad->ndim);

    int size = 1;
    for (int i = 0; i < output_grad->ndim; i++) {
        size *= output_grad->shape[i];
    }

    const float *output_grad_data = (const float *)output_grad->data;
    float *input_data = (float *)input->data;
    float *input_grad_data = (float *)input_grad->data;

    dim3 threads;
    dim3 blocks;

    if (size <= 1024) {
        blocks.x = 1;
        threads.x = size;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }

    if (stream_handle) {
        log_grad_kernel<<<blocks, threads, 0,
                          *(cudaStream_t *)stream_handle->handle>>>(
            output_grad_data, input_data, input_grad_data, eps, size);
    } else {
        log_grad_kernel<<<blocks, threads>>>(output_grad_data, input_data,
                                             input_grad_data, eps, size);
    }
    return 0;
}
