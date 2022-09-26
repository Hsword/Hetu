#include "gpu_reduce.h"

__global__ void reduce_sum_kernel(const float *input, float *output,
                                  int ndim_input, int ndim_rest, int ndim_reduce,
                                  size_t *strides, size_t *strides_reduce, size_t* stride_rest,
                                  size_t *shape_in, size_t *shape_rest, size_t *shape_reduce,
                                  int* reduce_dims, int* rest_dims, int reduce_num) {
    __shared__ float shared_sum[32];

    size_t start_index = threadIdx.x;
    size_t end_index = reduce_num;
    size_t output_ptr = blockIdx.x;
    if(start_index >= end_index)
        return ;

    size_t ptr_fix = 0, tmp = output_ptr, k;
    for(int i = 0; i < ndim_rest; ++i) {
        k = tmp / stride_rest[i];
        ptr_fix += k * strides[rest_dims[i]];
        tmp -= k * stride_rest[i];
    }

    float sum_thread = 0;
	for (size_t i = start_index, ptr; i < end_index; i += blockDim.x) {
        ptr = ptr_fix, tmp = i;
        for(int j = 0; j < ndim_reduce; ++j) {
            k = tmp / strides_reduce[j];
            ptr += k * strides[reduce_dims[j]];
            tmp -= k * strides_reduce[j];
        }
        sum_thread += input[ptr];
    }

    BlockReduceSum(sum_thread, shared_sum);
    if (threadIdx.x == 0)
        output[output_ptr] = sum_thread;
}

__global__ void reduce_sum_single_kernel(const float *input, float *output, size_t befor_dim_size,\
                                        size_t reduce_dim_size, size_t after_dim_size) {
    __shared__ float shared_sum[32];

    size_t x = blockIdx.x / after_dim_size;
    size_t y = blockIdx.x % after_dim_size;
    size_t start_ptr, end_ptr, stride;
    if (after_dim_size > 1) {
        stride = after_dim_size * blockDim.x;
        start_ptr = x * reduce_dim_size * after_dim_size + y + threadIdx.x * after_dim_size;
        end_ptr = x * reduce_dim_size * after_dim_size + y + reduce_dim_size * after_dim_size;
    }
    else {
        size_t cols_per_thread = (reduce_dim_size + blockDim.x - 1) / blockDim.x;
        size_t block_end_ptr = x * reduce_dim_size * after_dim_size + y + reduce_dim_size * after_dim_size;
        start_ptr = x * reduce_dim_size * after_dim_size + y + threadIdx.x * cols_per_thread * after_dim_size;
        end_ptr = min(start_ptr + cols_per_thread * after_dim_size, block_end_ptr);
        stride = after_dim_size;
    }
    size_t output_ptr = x * after_dim_size + y;
    if(start_ptr >= end_ptr)
        return ;

    float sum_thread = 0;
	for (size_t ptr = start_ptr; ptr < end_ptr; ptr += stride)
        sum_thread += input[ptr];

    BlockReduceSum(sum_thread, shared_sum);
    if (threadIdx.x == 0)
        output[output_ptr] = sum_thread;
}

int DLGpuReduceSum(const DLArrayHandle in_arr, DLArrayHandle out_arr, int *axes,
                   int num_ax, DLStreamHandle stream_handle = NULL) {
    if(num_ax <= 0)
        return 0;
    for(int i = 0 ; i < num_ax; ++i)
        assert(axes[i] >= 0 && axes[i] < in_arr->ndim);
    std::sort(axes, axes + num_ax);
    num_ax = std::unique(axes, axes + num_ax) - axes;

    int *reduce_dims = (int *)malloc(num_ax * sizeof(int));
    int *rest_dims = (int *)malloc((in_arr->ndim - num_ax) * sizeof(int));

    size_t *shape_in = (size_t *)malloc(in_arr->ndim * sizeof(size_t));
    size_t *shape_reduce = (size_t *)malloc(num_ax * sizeof(size_t));
    size_t *shape_rest = (size_t *)malloc((in_arr->ndim - num_ax) * sizeof(size_t));

    // merge continuous reduce_dims
    int reduce_num = 1, rest_num = 1;
    int ndim_input = 0, ndim_reduce = 0, ndim_rest = 0;
    for(int i = 0, p = 0; i < in_arr->ndim; ) {
        while (p < num_ax && axes[p] < i) 
            ++p;
        if(p < num_ax && axes[p] == i) {
            int reduce_size = 1;
            for (;p < num_ax && axes[p] == i; ++i, ++p)
                reduce_size *= in_arr->shape[i];
            reduce_dims[ndim_reduce] = ndim_input;
            shape_reduce[ndim_reduce++] = reduce_size;
            shape_in[ndim_input++] = reduce_size;
            reduce_num *= reduce_size;
        }
        else {
            rest_dims[ndim_rest] = ndim_input;
            shape_rest[ndim_rest++] = in_arr->shape[i];
            shape_in[ndim_input++] = in_arr->shape[i];
            rest_num *= in_arr->shape[i];
            ++i;
        }
    }

    if (ndim_reduce == 1) {
        size_t befor_dim_size, reduce_dim_size, after_dim_size;
        befor_dim_size = reduce_dim_size = after_dim_size = 1;
        for (int i = 0; i < ndim_input; ++i) {
            if(i < reduce_dims[0]) befor_dim_size *= shape_in[i];
            else if (i == reduce_dims[0]) reduce_dim_size = shape_in[i];
            else after_dim_size *= shape_in[i];
        }
        const float *input_data = (const float *)in_arr->data;
        float *output_data = (float *)out_arr->data;

        int blocks = befor_dim_size * after_dim_size;
        int threads = GetThreadNum(reduce_dim_size);
        if (stream_handle)
            reduce_sum_single_kernel<<<blocks, threads, 0,
                            *(cudaStream_t *)stream_handle->handle>>>(
                input_data, output_data, befor_dim_size, reduce_dim_size, after_dim_size);
        else
            reduce_sum_single_kernel<<<blocks, threads>>>(input_data, output_data, befor_dim_size, reduce_dim_size, after_dim_size);
    }
    else {
        size_t *strides = (size_t *)malloc(ndim_input * sizeof(size_t));
        size_t *strides_rest = (size_t *)malloc(ndim_rest * sizeof(size_t));
        size_t *strides_reduce = (size_t *)malloc(ndim_reduce * sizeof(size_t));

        strides[ndim_input - 1] = strides_reduce[ndim_reduce - 1] = strides_rest[ndim_rest - 1] = 1;
        for(int i = ndim_input - 2; i >= 0; --i)
            strides[i] = strides[i + 1] * shape_in[i + 1];
        for(int i = ndim_reduce - 2; i >= 0; --i)
            strides_reduce[i] = strides_reduce[i + 1] * shape_reduce[i + 1];
        for(int i = ndim_rest - 2; i >= 0; --i)
            strides_rest[i] = strides_rest[i + 1] * shape_rest[i + 1];

        int *reduce_dims_cu, *rest_dims_cu;
        cudaMalloc(&reduce_dims_cu, ndim_reduce * sizeof(int));
        cudaMalloc(&rest_dims_cu, ndim_rest * sizeof(int));
        cudaMemcpy(reduce_dims_cu, reduce_dims, ndim_reduce * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(rest_dims_cu, rest_dims, ndim_rest * sizeof(int), cudaMemcpyHostToDevice);

        size_t *shape_in_cu, *shape_rest_cu, *shape_reduce_cu;
        size_t *strides_cu, *strides_reduce_cu, *strides_rest_cu;\
        cudaMalloc(&shape_in_cu, ndim_input * sizeof(size_t));
        cudaMalloc(&shape_reduce_cu, ndim_reduce * sizeof(size_t));
        cudaMalloc(&shape_rest_cu, ndim_rest * sizeof(size_t));
        cudaMalloc(&strides_cu, ndim_input * sizeof(size_t));
        cudaMalloc(&strides_reduce_cu, ndim_reduce * sizeof(size_t));
        cudaMalloc(&strides_rest_cu, ndim_rest * sizeof(size_t));

        cudaMemcpy(shape_in_cu, shape_in, ndim_input * sizeof(size_t), cudaMemcpyHostToDevice);
        cudaMemcpy(shape_rest_cu, shape_rest, ndim_rest * sizeof(size_t), cudaMemcpyHostToDevice);
        cudaMemcpy(shape_reduce_cu, shape_reduce, ndim_reduce * sizeof(size_t), cudaMemcpyHostToDevice);
        cudaMemcpy(strides_cu, strides, ndim_input * sizeof(size_t), cudaMemcpyHostToDevice);
        cudaMemcpy(strides_reduce_cu, strides_reduce, ndim_reduce * sizeof(size_t), cudaMemcpyHostToDevice);
        cudaMemcpy(strides_rest_cu, strides_rest, ndim_rest * sizeof(size_t), cudaMemcpyHostToDevice);

        int blocks = rest_num;
        int threads = GetThreadNum(reduce_num);
        if (stream_handle)
            reduce_sum_kernel<<<blocks, threads, 0,
                            *(cudaStream_t *)stream_handle->handle>>>((float*)in_arr->data, (float*)out_arr->data,
                                                                    ndim_input, ndim_rest, ndim_reduce,
                                                                    strides_cu, strides_reduce_cu, strides_rest_cu,
                                                                    shape_in_cu, shape_rest_cu, shape_reduce_cu,
                                                                    reduce_dims_cu, rest_dims_cu, reduce_num);
        else
            reduce_sum_kernel<<<blocks, threads>>>((float*)in_arr->data, (float*)out_arr->data,
                                                ndim_input, ndim_rest, ndim_reduce,
                                                strides_cu, strides_reduce_cu, strides_rest_cu,
                                                shape_in_cu, shape_rest_cu, shape_reduce_cu,
                                                reduce_dims_cu, rest_dims_cu, reduce_num);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            printf("CUDA Error: %s\n", cudaGetErrorString(err));

        free(strides);
        free(strides_reduce);
        cudaFree(rest_dims_cu);
        cudaFree(reduce_dims_cu);
        cudaFree(shape_in_cu);
        cudaFree(shape_rest_cu);
        cudaFree(shape_reduce_cu);
        cudaFree(strides_cu);
        cudaFree(strides_rest_cu);
        cudaFree(strides_reduce_cu);
    }
    free(rest_dims);
    free(reduce_dims);
    free(shape_in);
    free(shape_rest);
    free(shape_reduce);
    return 0;
}
