#include "gpu_runtime.h"

__global__ void argmax_kernel(const float *input, int *output,
                              size_t befor_dim_size, size_t reduce_dim_size,
                              size_t after_dim_size) {
    size_t ind_x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t ind_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (ind_x >= befor_dim_size || ind_y >= after_dim_size) {
        return;
    }
    size_t start_ptr = ind_x * reduce_dim_size * after_dim_size + ind_y;
    size_t output_ptr = ind_x * after_dim_size + ind_y;

    int max_index = 0;
    float max_value = input[start_ptr];
    for (size_t s = 1; s < reduce_dim_size; ++s) {
        size_t cur_ind = start_ptr + s * after_dim_size;
        if (input[cur_ind] > max_value) {
            max_value = input[cur_ind];
            max_index = s;
        }
    }
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

    dim3 blocks;
    dim3 threads;
    ThreadBlock2D(threads, blocks, befor_dim_size, after_dim_size);
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