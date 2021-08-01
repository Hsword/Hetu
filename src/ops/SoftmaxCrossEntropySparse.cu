#include "gpu_runtime.h"

// y = inputs[0], y_ = inputs[1]
// np.mean(-np.sum(y_ * np.log(softmax(y)), axis=1), keepdims=True)
__global__ void matrix_softmax_cross_entropy_kernel(int nrow, int ncol,
                                                    const float *input_a,
                                                    const float *input_b,
                                                    const int ignored_index,                                                 
                                                    float *output) {
    // Two dimensional thread blocks.
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= nrow) return;
    if(int(input_b[id])==ignored_index)
    {
        output[id] = 0;
        return;
    }
            
    float maxval = input_a[id * ncol];
    // Find max for a row.
    for (int x = 1; x < ncol; ++x) {
        maxval = max(maxval, input_a[id * ncol + x]);
    }
    // Deduct by max for a row, and raise to exp.
    float sum = 0;
    for (int x = 0; x < ncol; ++x) {
        sum += exp(input_a[id * ncol + x] - maxval);
    }
    // Compute per-row loss.
    size_t curid = id * ncol + int(input_b[id]);
    float loss = -(input_a[curid] - maxval) + log(sum);
    output[id] = loss;
}

int DLGpuSoftmaxCrossEntropySparse(const DLArrayHandle input_a,
                             const DLArrayHandle input_b,
                             const int ignored_index,
                             DLArrayHandle output, DLStreamHandle stream_handle = NULL) {
    size_t indim = input_a->ndim;
    
    assert (output->ndim == input_b->ndim && indim == output->ndim + 1);
    int nrow = 1;
    for (int i = 0; i < indim-1; ++i) {
        assert (input_a->shape[i] == input_b->shape[i] &&
            input_a->shape[i] == output->shape[i]);
        nrow *= input_a->shape[i];
    }

    int ncol = input_a->shape[indim-1];

    const float *input_data_a = (const float *)input_a->data;
    const float *input_data_b = (const float *)input_b->data;
    float *output_data = (float *)output->data;

    dim3 blocks;
    dim3 threads;
    if (nrow <= 1024) {
        threads.x = nrow;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (nrow + 1023) / 1024;
    }
    // 1 block
    if (stream_handle) {
        matrix_softmax_cross_entropy_kernel<<<blocks, threads, 0, *(cudaStream_t*)stream_handle->handle>>>(
            nrow, ncol, input_data_a, input_data_b, ignored_index, output_data);
    } else {
        matrix_softmax_cross_entropy_kernel<<<blocks, threads>>>(
            nrow, ncol, input_data_a, input_data_b, ignored_index, output_data);
    }
    return 0;
}

__global__ void softmax_cross_entropy_gradient_kernel(int nrow, int ncol, const float *input_a, const float *input_b, const float *input_c, const int ignored_index, float *output) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= nrow) return;

    if(int(input_b[id])==ignored_index)
    {
        for (int x = 0; x < ncol; ++x) 
        {
            size_t curid = id * ncol + x;
            output[curid] = 0;
        }
        return;        
    }
    
    float maxval = input_a[id * ncol];
    // Find max for a row.
    for (int x = 1; x < ncol; ++x) {
        maxval = max(maxval, input_a[id * ncol + x]);
    }
    // Deduct by max for a row, and raise to exp.
    float sum = 0;
    for (int x = 0; x < ncol; ++x) {
        sum += exp(input_a[id * ncol + x] - maxval);
    }
    for (int x = 0; x < ncol; ++x) {
        size_t curid = id * ncol + x;
        if(x == int(input_b[id]))
        {
            output[curid] = (exp(input_a[curid] - maxval) / sum - 1.0) * input_c[id];
        }
        else
            output[curid] = (exp(input_a[curid] - maxval) / sum) * input_c[id];
    }
}

int DLGpuSoftmaxCrossEntropySparse_Gradient(const DLArrayHandle input_a, const DLArrayHandle input_b,
                                      const DLArrayHandle input_c, const int ignored_index,
                                      DLArrayHandle output,
                                      DLStreamHandle stream_handle = NULL) {
    size_t indim = input_a->ndim;
    assert (indim >= 2 && input_c->ndim == input_b->ndim && indim == input_c->ndim + 1 && indim == output->ndim);
    int nrow = 1;
    for (int i = 0; i < indim-1; ++i) {
        assert (input_a->shape[i] == input_b->shape[i] &&
            input_a->shape[i] == output->shape[i] &&
            input_a->shape[i] == input_c->shape[i]);
        nrow *= input_a->shape[i];
    }
    assert (input_a->shape[indim-1] == output->shape[indim-1]);
    int ncol = input_a->shape[indim-1];
        
    const float *input_data_a = (const float *)input_a->data;
    const float *input_data_b = (const float *)input_b->data;
    const float *input_data_c = (const float *)input_c ->data;   
    float *output_data = (float *)output->data;

    dim3 blocks;
    dim3 threads;
    if (nrow <= 1024) {
        threads.x = nrow;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (nrow + 1023) / 1024;
    }
    if (stream_handle) {
        softmax_cross_entropy_gradient_kernel<<<blocks, threads, 0, *(cudaStream_t*)stream_handle->handle>>>(
            nrow, ncol, input_data_a, input_data_b, input_data_c, ignored_index, output_data);
    } else {
        softmax_cross_entropy_gradient_kernel<<<blocks, threads>>>(
            nrow, ncol, input_data_a, input_data_b, input_data_c, ignored_index, output_data);
    }
    return 0;
}
