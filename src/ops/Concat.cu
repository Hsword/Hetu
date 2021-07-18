#include "gpu_runtime.h"

__global__ void concat_kernel(const float *x_data, const float *y_data,
                              float *output_data, int concat_size, int offset1,
                              int offset2, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    int all_offset = offset1 + offset2;
    int post_ind = ind % concat_size;
    int temp = ind / concat_size;
    int mid_ind = temp % all_offset;
    int pre_ind = temp / all_offset;
    float val;
    if (mid_ind < offset1) {
        int x_ind = (pre_ind * offset1 + mid_ind) * concat_size + post_ind;
        val = x_data[x_ind];
    } else {
        int y_ind =
            (pre_ind * offset2 + mid_ind - offset1) * concat_size + post_ind;
        val = y_data[y_ind];
    }
    output_data[ind] = val;
}

int DLGpuConcat(const DLArrayHandle input_x, const DLArrayHandle input_y,
                DLArrayHandle output, int axis = 0,
                DLStreamHandle stream_handle = NULL) {
    int now_ndim = input_x->ndim;
    assert(axis >= 0 && axis < now_ndim);
    assert(now_ndim == input_y->ndim && now_ndim == output->ndim);
    int num_concats = 1;
    for (int i = 0; i < axis; ++i) {
        int cur_dim = input_x->shape[i];
        assert(cur_dim == input_y->shape[i] && cur_dim == output->shape[i]);
        num_concats *= cur_dim;
    }
    int offset1 = input_x->shape[axis];
    int offset2 = input_y->shape[axis];
    assert(offset1 + offset2 == output->shape[axis]);
    int concat_size = 1;
    for (int i = axis + 1; i < now_ndim; i++) {
        int cur_dim = input_x->shape[i];
        assert(cur_dim == input_y->shape[i] && cur_dim == output->shape[i]);
        concat_size *= cur_dim;
    }
    size_t size = num_concats * (offset1 + offset2) * concat_size;

    const float *x_data = (const float *)(input_x->data);
    const float *y_data = (const float *)(input_y->data);
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
        concat_kernel<<<blocks, threads, 0,
                        *(cudaStream_t *)stream_handle->handle>>>(
            x_data, y_data, output_data, concat_size, offset1, offset2, size);
    } else {
        concat_kernel<<<blocks, threads>>>(x_data, y_data, output_data,
                                           concat_size, offset1, offset2, size);
    }
    return 0;
}

__global__ void concat_gradient_kernel(const float *o_data, float *i_data,
                                       int concat_size, int concat_offset,
                                       int small_offset, int big_offset,
                                       size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    int post_ind = ind % concat_size;
    int temp = ind / concat_size;
    int mid_ind = temp % small_offset + concat_offset;
    int pre_ind = temp / small_offset;
    int o_ind = (pre_ind * big_offset + mid_ind) * concat_size + post_ind;
    i_data[ind] = o_data[o_ind];
}

int DLGpuConcat_gradient(const DLArrayHandle o_grad, DLArrayHandle i_grad,
                         int axis = 0, int id = 0,
                         DLStreamHandle stream_handle = NULL) {
    int now_ndim = o_grad->ndim;
    assert(now_ndim == i_grad->ndim);
    int num_concats = 1;
    for (int i = 0; i < axis; ++i) {
        int cur_dim = o_grad->shape[i];
        assert(cur_dim == i_grad->shape[i]);
        num_concats *= cur_dim;
    }
    int big_offset = o_grad->shape[axis];
    int small_offset = i_grad->shape[axis];
    int concat_offset = (id == 1) ? (big_offset - small_offset) : 0;
    int concat_size = 1;
    for (int i = axis + 1; i < now_ndim; ++i) {
        int cur_dim = o_grad->shape[i];
        assert(cur_dim == i_grad->shape[i]);
        concat_size *= cur_dim;
    }
    const float *o_data = (const float *)(o_grad->data);
    float *i_data = (float *)(i_grad->data);
    size_t size = num_concats * small_offset * concat_size;

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
        concat_gradient_kernel<<<blocks, threads, 0,
                                 *(cudaStream_t *)stream_handle->handle>>>(
            o_data, i_data, concat_size, concat_offset, small_offset,
            big_offset, size);
    } else {
        concat_gradient_kernel<<<blocks, threads>>>(o_data, i_data, concat_size,
                                                    concat_offset, small_offset,
                                                    big_offset, size);
    }
    return 0;
}
