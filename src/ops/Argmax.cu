#include "gpu_reduce.h"

__global__ void argmax_kernel(const float *input, int *output, size_t befor_dim_size,\
                                        size_t reduce_dim_size, size_t after_dim_size) {
    __shared__ size_t shared_max_ptr[32];
    __shared__ float shared_max_value[32];

    // block dim3 is slower than using dim1 
    size_t x = blockIdx.x / after_dim_size;
    size_t y = blockIdx.x % after_dim_size;
    size_t stride = after_dim_size * blockDim.x;
    size_t start_ptr = x * reduce_dim_size * after_dim_size + y + threadIdx.x * after_dim_size;
    size_t end_ptr = x * reduce_dim_size * after_dim_size + y + reduce_dim_size * after_dim_size;
    size_t output_ptr = x * after_dim_size + y;
    if(start_ptr >= end_ptr)
        return ;

    size_t max_index = threadIdx.x;
    float max_value = input[start_ptr];

	for (size_t i = threadIdx.x + blockDim.x, ptr = start_ptr + stride; ptr < end_ptr; i += blockDim.x, ptr += stride) {
        if (input[ptr] > max_value) {
            max_value = input[ptr];
            max_index = i;
        }
	}

    BlockReduceArgmax(max_value, max_index, shared_max_value, shared_max_ptr);
    if (threadIdx.x == 0)
        output[output_ptr] = max_index;
}

int DLGpuArgmax(const DLArrayHandle input, DLArrayHandle output, int dim,
                DLStreamHandle stream_handle) {
    assert(input->ndim == output->ndim + 1);
    size_t befor_dim_size, reduce_dim_size, after_dim_size;
    befor_dim_size = reduce_dim_size = after_dim_size = 1;
    for (int i = 0; i < input->ndim; ++i) {
        if (i < dim)
            befor_dim_size *= input->shape[i];
        else if (i == dim)
            reduce_dim_size = input->shape[i];
        else
            after_dim_size *= input->shape[i];
    }
    const float *input_data = (const float *)input->data;
    int *output_data = (int *)output->data;

    int blocks = befor_dim_size * after_dim_size;
    int threads = GetThreadNum(reduce_dim_size);
    if (stream_handle)
        argmax_kernel<<<blocks, threads, 0,
                        *(cudaStream_t *)stream_handle->handle>>>(
            input_data, output_data, befor_dim_size, reduce_dim_size,
            after_dim_size);
    else
        argmax_kernel<<<blocks, threads>>>(input_data, output_data,
                                           befor_dim_size, reduce_dim_size,
                                           after_dim_size);
    return 0;
}
