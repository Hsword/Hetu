#include "gpu_runtime.h"

__global__ void argmax_kernel(const float *input, float *output, size_t befor_dim_size,\
                                        size_t reduce_dim_size, size_t after_dim_size) {
    size_t ind_x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t ind_y = blockIdx.y * blockDim.y + threadIdx.y;
    if(ind_x >= befor_dim_size || ind_y >= after_dim_size){
        return;
    }
    size_t start_ptr = ind_x * reduce_dim_size * after_dim_size + ind_y;
    // size_t offset = after_dim_size;
    // size_t offset_cnt = reduce_dim_size;
    
    size_t output_ptr = ind_x * after_dim_size + ind_y;
    
    // // Shared Memory for dynamic array
    // extern __shared__ int s[];
    // int *maxindexes = s;
	// double *maxvalues = (double*)&maxindexes[blockDim.x*blockDim.y];

	// maxindexes[i] = start_ptr;
	// maxvalues[i] = input[start_ptr];

	// __syncthreads();
    float max_index = start_ptr;
    size_t max_value = input[start_ptr];
	// do reduction in shared mem
	for (size_t s = 1; s < reduce_dim_size; s += 1)
	{
        if (input[start_ptr + s * after_dim_size] > max_value){
            max_value = input[start_ptr + s * after_dim_size];
            max_index = start_ptr + s * after_dim_size;
        }
		// __syncthreads();
	}
    output[output_ptr] = max_index;
}

int DLGpuArgmax(const DLArrayHandle input, DLArrayHandle output, int dim,
                DLStreamHandle stream_handle) {
    assert(input->ndim == output->ndim - 1);
    size_t befor_dim_size, reduce_dim_size, after_dim_size;
    befor_dim_size = reduce_dim_size = after_dim_size = 1;
    for (int i = 0; i < input->ndim; ++i) {
        if(i < dim) befor_dim_size *= input->shape[i];
        else if (i == dim) reduce_dim_size = input->shape[i];
        else after_dim_size *= input->shape[i];
    }
    const float *input_data = (const float *)input->data;
    float *output_data = (float *)output->data;

    dim3 blocks;
    dim3 threads;
    if (befor_dim_size <= 1024) {
        threads.x = befor_dim_size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (befor_dim_size + 1023) / 1024;
    }
    if (after_dim_size <= 1024) {
        threads.y = after_dim_size;
        blocks.y = 1;
    } else {
        threads.y = 1024;
        blocks.y = (after_dim_size + 1023) / 1024;
    }
    if (stream_handle)
        argmax_kernel<<<blocks, threads, 0,
                        *(cudaStream_t *)stream_handle->handle>>>(
            input_data, output_data, befor_dim_size, reduce_dim_size, after_dim_size);
    else
        argmax_kernel<<<blocks, threads>>>(input_data, output_data, befor_dim_size, reduce_dim_size, after_dim_size);
    return 0;
}