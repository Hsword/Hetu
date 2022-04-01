#include "gpu_runtime.h"

__global__ void cumsum_with_bias(const float* input_data, float* output_data, int before_dim,\
                                    int this_dim, int after_dim, float bias){
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= before_dim * after_dim)
        return;
    int x = ind / after_dim;
    int y = ind % after_dim;
    int offset = after_dim;
    const float *input_ptr = input_data + x * this_dim * after_dim + y;
    float *output_ptr = output_data + x * this_dim * after_dim + y;
    float pre_sum = 0;
    for(int i = 0; i < this_dim; i++){
        pre_sum += *(input_ptr + offset * i);
        *(output_ptr + offset * i) = pre_sum - 1;
    }
}

int DLGpuCumsumWithBias(DLArrayHandle input, DLArrayHandle output, float bias, int dim, DLStreamHandle stream_handle) {
    size_t before_dim, this_dim, after_dim;
    before_dim = after_dim = 1;
    for (index_t i = 0; i < input->ndim; i++) {
        if (i < dim){
            before_dim = before_dim * (input->shape[i]);
        }
        else if(i == dim){
            this_dim = input->shape[i];
        }
        else{
            after_dim = after_dim * (input->shape[i]);
        }
    }
    size_t size = before_dim * after_dim;
    dim3 blocks;
    dim3 threads;
    const float *input_data = (const float *)input->data;
    float *output_data = (float *)output->data;
    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }
    if (stream_handle)
        cumsum_with_bias<<<blocks, threads, 0, *(cudaStream_t *)stream_handle->handle>>> \
            (input_data, output_data, before_dim, this_dim, after_dim, bias);
    else
        cumsum_with_bias<<<blocks, threads>>>(input_data, output_data, before_dim, \
            this_dim, after_dim, bias);
    return 0;
}
    