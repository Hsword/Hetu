#include "gpu_runtime.h"

__global__ void cross_entropy_sparse_kernel(const float *y,
                                            const float *y_, float *output,
                                            const int ignored_index,
                                            int nrow, int ncol) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nrow)
        return;
    int pos = int(y_[idx]);
    if(pos == ignored_index || pos < 0){
        output[idx] = 0;
        return;
    }
    int y_pos = idx * ncol + pos;
    output[idx] = -log(y[y_pos]);
}

int DLGpuCrossEntropySparse(const DLArrayHandle input_y,
                              const DLArrayHandle label, 
                              const int ignored_index,
                              DLArrayHandle output,
                              DLStreamHandle stream_handle = NULL) {
    size_t indim = input_y->ndim;
    assert(indim == label->ndim + 1 && indim == output->ndim + 1);
    size_t nrow = 1;
    for (int i = 0; i < indim - 1; ++i) {
        nrow *= input_y->shape[i];
    }
    int ncol = input_y->shape[indim - 1];
    size_t size = nrow;
    const float *y_data = (const float *)(input_y->data);
    const float *label_data = (const float *)(label->data);
    float *output_data = (float *)(output->data);

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
        cross_entropy_sparse_kernel<<<blocks, threads, 0,
                                     *(cudaStream_t *)stream_handle->handle>>>(
            y_data, label_data, output_data, ignored_index, nrow, ncol);
    } else {
        cross_entropy_sparse_kernel<<<blocks, threads>>>(
            y_data, label_data, output_data, ignored_index, nrow, ncol);
    }
    return 0;
}

__global__ void cross_entropy_sparse_gradient_kernel(const float *y, const float *y_,
                                     const float *grad_data, float *output_data,
                                     const int ignored_index, int length, size_t size) {
    size_t thread_ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_ind >= size)
        return;
    size_t ind = thread_ind / length;
    size_t offset = thread_ind % length;
    int pos = int(y_[ind]);
    if(pos == ignored_index || pos < 0){
        output_data[thread_ind] = 0;
        return;
    }
    if(pos != offset){
        output_data[thread_ind] = 0;
        return;
    }
    output_data[thread_ind] = - 1.0 / y[thread_ind] * grad_data[ind];
}

int DLGpuCrossEntropySparseGradient(const DLArrayHandle grad,
                                      const DLArrayHandle input_y,
                                      const DLArrayHandle label,
                                      const int ignored_index,
                                      DLArrayHandle output,
                                      DLStreamHandle stream_handle = NULL) {
    size_t indim = input_y->ndim;
    assert(indim == label->ndim + 1 && indim == output->ndim
           && indim == grad->ndim + 1);

    int nrow = 1;
    for (int i = 0; i < indim - 1; ++i) {
        nrow *= input_y->shape[i];
    }
    int ncol = input_y->shape[indim - 1];
    size_t size = nrow * ncol;
    const float *grad_data = (const float *)grad->data;
    const float *y_data = (const float *)input_y->data;
    const float *label_data = (const float *)label->data;
    float *output_data = (float *)output->data;

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
        cross_entropy_sparse_gradient_kernel<<<
            blocks, threads, 0, *(cudaStream_t *)stream_handle->handle>>>(
            y_data, label_data, grad_data, output_data, ignored_index, ncol,
            size);
    } else {
        cross_entropy_sparse_gradient_kernel<<<blocks, threads>>>(
            y_data, label_data, grad_data, output_data, ignored_index, ncol,
            size);
    }
    return 0;
}
