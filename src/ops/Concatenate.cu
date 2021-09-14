#include "gpu_runtime.h"

__global__ void concatenate_kernel(const float *input_data, float *output_data,
                                   int input_width, int output_width,
                                   int offset, int concat_size, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    int post_ind = ind % concat_size;
    int prev_ind = ind / concat_size;
    int mid_ind = prev_ind % input_width + offset;
    prev_ind = prev_ind / input_width;
    int out_ind = (prev_ind * output_width + mid_ind) * concat_size + post_ind;
    output_data[out_ind] = input_data[ind];
}

int DLGpuConcatenate(const DLArrayHandle input, DLArrayHandle output, int axis,
                     int offset, DLStreamHandle stream_handle = NULL) {
    int now_ndim = output->ndim;
    assert(input->ndim == now_ndim);
    int num_concats = 1;
    for (int i = 0; i < axis; ++i) {
        int cur_dim = output->shape[i];
        assert(input->shape[i] == cur_dim);
        num_concats *= cur_dim;
    }
    int concat_size = 1;
    for (int i = axis + 1; i < now_ndim; ++i) {
        int cur_dim = output->shape[i];
        assert(input->shape[i] == cur_dim);
        concat_size *= cur_dim;
    }
    const float *input_data = (const float *)(input->data);
    float *output_data = (float *)(output->data);
    int input_width = input->shape[axis];
    int output_width = output->shape[axis];

    dim3 blocks;
    dim3 threads;
    size_t size = num_concats * input_width * concat_size;
    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }

    if (stream_handle) {
        concatenate_kernel<<<blocks, threads, 0,
                             *(cudaStream_t *)stream_handle->handle>>>(
            input_data, output_data, input_width, output_width, offset,
            concat_size, size);
    } else {
        concatenate_kernel<<<blocks, threads>>>(input_data, output_data,
                                                input_width, output_width,
                                                offset, concat_size, size);
    }
    return 0;
}

__global__ void concatenate_gradient_kernel(const float *o_data, float *i_data,
                                            int input_width, int output_width,
                                            int offset, int concat_size,
                                            size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    int post_ind = ind % concat_size;
    int prev_ind = ind / concat_size;
    int mid_ind = prev_ind % input_width + offset;
    prev_ind = prev_ind / input_width;
    int out_ind = (prev_ind * output_width + mid_ind) * concat_size + post_ind;
    i_data[ind] = o_data[out_ind];
}

int DLGpuConcatenate_gradient(const DLArrayHandle o_grad, DLArrayHandle i_grad,
                              int axis, int offset,
                              DLStreamHandle stream_handle = NULL) {
    int now_ndim = o_grad->ndim;
    assert(now_ndim == i_grad->ndim);
    int num_concats = 1;
    for (int i = 0; i < axis; ++i) {
        int cur_dim = o_grad->shape[i];
        assert(cur_dim == i_grad->shape[i]);
        num_concats *= cur_dim;
    }
    int concat_size = 1;
    for (int i = axis + 1; i < now_ndim; ++i) {
        int cur_dim = o_grad->shape[i];
        assert(cur_dim == i_grad->shape[i]);
        concat_size *= cur_dim;
    }
    const float *o_data = (const float *)(o_grad->data);
    float *i_data = (float *)(i_grad->data);
    int output_width = o_grad->shape[axis];
    int input_width = i_grad->shape[axis];

    dim3 blocks;
    dim3 threads;
    size_t size = num_concats * input_width * concat_size;
    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }

    if (stream_handle) {
        concatenate_gradient_kernel<<<blocks, threads, 0,
                                      *(cudaStream_t *)stream_handle->handle>>>(
            o_data, i_data, input_width, output_width, offset, concat_size,
            size);
    } else {
        concatenate_gradient_kernel<<<blocks, threads>>>(
            o_data, i_data, input_width, output_width, offset, concat_size,
            size);
    }
    return 0;
}
