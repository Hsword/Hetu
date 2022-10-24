#include "gpu_reduce.h"

__global__ void p_norm_kernel(const float *input, float *output, int p,
                              size_t befor_dim_size, size_t reduce_dim_size,
                              size_t after_dim_size) {
    __shared__ float shared_sum[32];

    size_t x = blockIdx.x / after_dim_size;
    size_t y = blockIdx.x % after_dim_size;
    size_t start_ptr, end_ptr, stride;
    if (after_dim_size > 1) {
        stride = after_dim_size * blockDim.x;
        start_ptr = x * reduce_dim_size * after_dim_size + y
                    + threadIdx.x * after_dim_size;
        end_ptr = x * reduce_dim_size * after_dim_size + y
                  + reduce_dim_size * after_dim_size;
    } else {
        size_t cols_per_thread =
            (reduce_dim_size + blockDim.x - 1) / blockDim.x;
        size_t block_end_ptr = x * reduce_dim_size * after_dim_size + y
                               + reduce_dim_size * after_dim_size;
        start_ptr = x * reduce_dim_size * after_dim_size + y
                    + threadIdx.x * cols_per_thread * after_dim_size;
        end_ptr =
            min(start_ptr + cols_per_thread * after_dim_size, block_end_ptr);
        stride = after_dim_size;
    }
    size_t output_ptr = x * after_dim_size + y;
    if (start_ptr >= end_ptr)
        return;

    float sum_thread = 0;
    for (size_t ptr = start_ptr; ptr < end_ptr; ptr += stride)
        sum_thread += pow(abs(input[ptr]), p);

    BlockReduceSum(sum_thread, shared_sum);
    if (threadIdx.x == 0)
        output[output_ptr] = pow(sum_thread, 1.0 / p);
}

int DLGpuNorm(const DLArrayHandle in_arr, DLArrayHandle out_arr, int axis,
              int p, DLStreamHandle stream_handle = NULL) {
    size_t befor_dim_size, reduce_dim_size, after_dim_size;
    befor_dim_size = reduce_dim_size = after_dim_size = 1;
    for (int i = 0; i < in_arr->ndim; ++i) {
        if (i < axis)
            befor_dim_size *= in_arr->shape[i];
        else if (i == axis)
            reduce_dim_size = in_arr->shape[i];
        else
            after_dim_size *= in_arr->shape[i];
    }
    const float *input_data = (const float *)in_arr->data;
    float *output_data = (float *)out_arr->data;

    int blocks = befor_dim_size * after_dim_size;
    int threads = 1024;

    if (stream_handle)
        p_norm_kernel<<<blocks, threads, 0,
                        *(cudaStream_t *)stream_handle->handle>>>(
            input_data, output_data, p, befor_dim_size, reduce_dim_size,
            after_dim_size);
    else
        p_norm_kernel<<<blocks, threads>>>(input_data, output_data, p,
                                           befor_dim_size, reduce_dim_size,
                                           after_dim_size);

    return 0;
}

__device__ float sgn(float x) {
    if (x == 0.0)
        return 0.0;
    return x / abs(x);
}
__global__ void p_norm_gradient_kernel(const float *input, const float *norm,
                                       const float *grad, float *output, int p,
                                       size_t reduce_dim_size,
                                       size_t after_dim_size, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    int na = ind / (reduce_dim_size * after_dim_size);
    int nc = (ind % (reduce_dim_size * after_dim_size)) % after_dim_size;
    int ind_y = na * after_dim_size + nc;

    float input_val = input[ind];
    float grad_val = grad[ind_y];

    if (p == 1) {
        output[ind] = sgn(input_val) * grad_val;
    } else if (p == 2) {
        float norm_val = norm[ind_y];
        if (norm_val == 0)
            output[ind] = 0;
        else
            output[ind] = grad_val * input_val / norm_val;
    } else if (p > 2) {
        float norm_val = norm[ind_y];
        if (norm_val == 0)
            output[ind] = 0;
        else
            output[ind] = input_val * pow(abs(input_val), p - 2) * grad_val
                          / pow(norm_val, p - 1);
    }
}

int DLGpuNormGradient(const DLArrayHandle in_arr, const DLArrayHandle in_arr_y,
                      const DLArrayHandle grad_y, DLArrayHandle out_arr,
                      int axis, int p, DLStreamHandle stream_handle = NULL) {
    size_t reduce_dim_size, after_dim_size, size;
    reduce_dim_size = after_dim_size = size = 1;
    for (int i = 0; i < in_arr->ndim; ++i) {
        size *= in_arr->shape[i];
        if (i == axis)
            reduce_dim_size = in_arr->shape[i];
        else if (i > axis)
            after_dim_size *= in_arr->shape[i];
    }
    const float *in_arr_data = (const float *)in_arr->data;
    const float *in_arr_y_data = (const float *)in_arr_y->data;
    const float *grad_y_data = (const float *)grad_y->data;
    float *output_data = (float *)out_arr->data;

    dim3 blocks;
    dim3 threads;
    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }

    if (stream_handle)
        p_norm_gradient_kernel<<<blocks, threads, 0,
                                 *(cudaStream_t *)stream_handle->handle>>>(
            in_arr_data, in_arr_y_data, grad_y_data, output_data, p,
            reduce_dim_size, after_dim_size, size);
    else
        p_norm_gradient_kernel<<<blocks, threads>>>(
            in_arr_data, in_arr_y_data, grad_y_data, output_data, p,
            reduce_dim_size, after_dim_size, size);

    return 0;
}