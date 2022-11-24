#include "gpu_runtime.h"

__global__ void index_select_kernel(const float *input, const float *index,
                                    float *output, size_t in_row,
                                    size_t out_row, size_t col, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind > size) {
        return;
    }
    size_t a_id = ind / (out_row * col);
    size_t b_id = (ind % (out_row * col)) / col;
    size_t c_id = (ind % (out_row * col)) % col;
    output[ind] = input[a_id * in_row * col + int(index[b_id]) * col + c_id];
}

int DLGpuIndexSelect(const DLArrayHandle input, DLArrayHandle index,
                     DLArrayHandle output, int dim,
                     DLStreamHandle stream_handle = NULL) {
    assert(index->ndim == 1);
    size_t size = 1;
    size_t in_row = 1;
    size_t out_row = 1;
    size_t col = 1;

    for (int i = 0; i < output->ndim; i++) {
        if (i == dim) {
            in_row = input->shape[i];
            out_row = output->shape[i];
        } else if (i > dim) {
            col *= output->shape[i];
        }
        size *= output->shape[i];
    }

    const float *input_data = (const float *)input->data;
    float *index_data = (float *)index->data;
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
        index_select_kernel<<<blocks, threads, 0,
                              *(cudaStream_t *)stream_handle->handle>>>(
            input_data, index_data, output_data, in_row, out_row, col, size);
    } else {
        index_select_kernel<<<blocks, threads>>>(
            input_data, index_data, output_data, in_row, out_row, col, size);
    }
    return 0;
}

__global__ void index_select_grad_kernel(const float *grad, const float *index,
                                         float *output, size_t grad_row,
                                         size_t out_row, size_t col,
                                         size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind > size) {
        return;
    }
    float val = grad[ind];
    size_t a_id = ind / (grad_row * col);
    size_t b_id = (ind % (grad_row * col)) / col;
    size_t c_id = (ind % (grad_row * col)) % col;

    size_t ind_new = a_id * out_row * col + int(index[b_id]) * col + c_id;
    atomicAdd(&output[ind_new], val);
}

int DLGpuIndexSelectGrad(const DLArrayHandle grad, const DLArrayHandle index,
                         DLArrayHandle output, int dim,
                         DLStreamHandle stream_handle = NULL) {
    assert(index->ndim == 1);
    size_t size = 1;
    size_t grad_row = 1;
    size_t out_row = 1;
    size_t col = 1;

    for (int i = 0; i < output->ndim; i++) {
        if (i == dim) {
            grad_row = grad->shape[i];
            out_row = output->shape[i];
        } else if (i > dim) {
            col *= output->shape[i];
        }
        size *= grad->shape[i];
    }

    const float *grad_data = (const float *)grad->data;
    float *index_data = (float *)index->data;
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
        index_select_grad_kernel<<<blocks, threads, 0,
                                   *(cudaStream_t *)stream_handle->handle>>>(
            grad_data, index_data, output_data, grad_row, out_row, col, size);
    } else {
        index_select_grad_kernel<<<blocks, threads>>>(
            grad_data, index_data, output_data, grad_row, out_row, col, size);
    }
    return 0;
}