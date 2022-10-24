#include "gpu_runtime.h"

__global__ void gather_kernel(const float *input, const float *index,
                              float *output, size_t size, int dim_b_input,
                              int dim_c_input, int dim_b_output,
                              int dim_c_output) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    int na = ind / (dim_b_output * dim_c_output);
    int tmp = ind % (dim_b_output * dim_c_output);
    int nc = tmp % dim_c_output;
    int ind_new =
        na * dim_b_input * dim_c_input + int(index[ind]) * dim_c_input + nc;
    output[ind] = input[ind_new];
}

__global__ void gather_gradient_kernel(const float *input, const float *index,
                                       float *output, size_t size,
                                       int dim_b_input, int dim_c_input,
                                       int dim_b_output, int dim_c_output) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    float val = input[ind];
    int na = ind / (dim_b_output * dim_c_output);
    int tmp = ind % (dim_b_output * dim_c_output);
    int nc = tmp % dim_c_output;
    int ind_new =
        na * dim_b_input * dim_c_input + int(index[ind]) * dim_c_input + nc;
    atomicAdd(&output[ind_new], val);
}

int DLGpuGather(const DLArrayHandle input, const DLArrayHandle index,
                DLArrayHandle output, int dim,
                DLStreamHandle stream_handle = NULL) {
    size_t size = 1;
    int dim_b_input = input->shape[dim];
    int dim_c_input = 1;
    int dim_b_output = output->shape[dim];
    int dim_c_output = 1;

    for (index_t i = 0; i < input->ndim; i++) {
        size *= output->shape[i];
        if (i > dim) {
            dim_c_input *= input->shape[i];
            dim_c_output *= output->shape[i];
        }
    }

    dim3 blocks;
    dim3 threads;
    float *input_data = (float *)input->data;
    float *index_data = (float *)index->data;
    float *output_data = (float *)output->data;
    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }
    if (stream_handle)
        gather_kernel<<<blocks, threads, 0,
                        *(cudaStream_t *)stream_handle->handle>>>(
            input_data, index_data, output_data, size, dim_b_input, dim_c_input,
            dim_b_output, dim_c_output);
    else
        gather_kernel<<<blocks, threads>>>(input_data, index_data, output_data,
                                           size, dim_b_input, dim_c_input,
                                           dim_b_output, dim_c_output);
    return 0;
}

int DLGpuGatherGradient(const DLArrayHandle input, const DLArrayHandle index,
                        DLArrayHandle output, int dim,
                        DLStreamHandle stream_handle = NULL) {
    size_t size = 1;
    int dim_b_input = input->shape[dim];
    int dim_c_input = 1;
    int dim_b_output = output->shape[dim];
    int dim_c_output = 1;

    for (index_t i = 0; i < input->ndim; i++) {
        size *= input->shape[i];
        if (i > dim) {
            dim_c_input *= input->shape[i];
            dim_c_output *= output->shape[i];
        }
    }

    dim3 blocks;
    dim3 threads;
    float *input_data = (float *)input->data;
    float *index_data = (float *)index->data;
    float *output_data = (float *)output->data;
    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }
    if (stream_handle)
        gather_gradient_kernel<<<blocks, threads, 0,
                                 *(cudaStream_t *)stream_handle->handle>>>(
            input_data, index_data, output_data, size, dim_b_output,
            dim_c_output, dim_b_input, dim_c_input);
    else
        gather_gradient_kernel<<<blocks, threads>>>(
            input_data, index_data, output_data, size, dim_b_output,
            dim_c_output, dim_b_input, dim_c_input);
    return 0;
}