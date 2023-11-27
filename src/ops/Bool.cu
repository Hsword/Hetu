#include "gpu_runtime.h"

__global__ void bool_kernel(float *input, float *output, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    if(input[ind] > 0){
        output[ind] = 1;
    }
    else{
        output[ind] = 0;
    }
}

int DLGpuBool(const DLArrayHandle input, DLArrayHandle output,
              DLStreamHandle stream_handle = NULL) {
    size_t size = 1;
    for (index_t i = 0; i < input->ndim; i++) {
        size *= input->shape[i];
    }
    dim3 blocks;
    dim3 threads;
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;
    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }
    if (stream_handle)
        bool_kernel<<<blocks, threads, 0,
                      *(cudaStream_t *)stream_handle->handle>>>(
            input_data, output_data, size);
    else
        bool_kernel<<<blocks, threads>>>(input_data, output_data, size);
    return 0;
}

__global__ void bool_val_kernel(const float *input_A, float *output, float val,
                                int cond, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    if (cond == 0) {
        if (abs(input_A[ind] - val) < 1e-6)
            output[ind] = 1;
        else
            output[ind] = 0;
    } else if (cond == 1) {
        if (input_A[ind] - val < -1e-6)
            output[ind] = 1;
        else
            output[ind] = 0;
    } else if (cond == 2) {
        if (input_A[ind] - val > 1e-6)
            output[ind] = 1;
        else
            output[ind] = 0;
    } else if (cond == 3) {
        if (input_A[ind] - val < 1e-6)
            output[ind] = 1;
        else
            output[ind] = 0;
    } else if (cond == 4) {
        if (input_A[ind] - val > -1e-6)
            output[ind] = 1;
        else
            output[ind] = 0;
    }
}

int DLGpuBoolVal(const DLArrayHandle input, float val, DLArrayHandle output,
                 int cond, DLStreamHandle stream_handle = NULL) {
    size_t size = 1;
    for (index_t i = 0; i < input->ndim; i++) {
        size *= input->shape[i];
    }
    dim3 blocks;
    dim3 threads;
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;
    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }
    if (stream_handle)
        bool_val_kernel<<<blocks, threads, 0,
                          *(cudaStream_t *)stream_handle->handle>>>(
            input_data, output_data, val, cond, size);
    else
        bool_val_kernel<<<blocks, threads>>>(input_data, output_data, val, cond,
                                             size);
    return 0;
}

__global__ void bool_matrix_kernel(const float *input_A, const float *input_B,
                                   float *output, int cond, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    if (cond == 0) {
        if (abs(input_A[ind] - input_B[ind]) < 1e-6)
            output[ind] = 1;
        else
            output[ind] = 0;
    } else if (cond == 1) {
        if (input_A[ind] - input_B[ind] < -1e-6)
            output[ind] = 1;
        else
            output[ind] = 0;
    } else if (cond == 2) {
        if (input_A[ind] - input_B[ind] > 1e-6)
            output[ind] = 1;
        else
            output[ind] = 0;
    } else if (cond == 3) {
        if (input_A[ind] - input_B[ind] < 1e-6)
            output[ind] = 1;
        else
            output[ind] = 0;
    } else if (cond == 4) {
        if (input_A[ind] - input_B[ind] > -1e-6)
            output[ind] = 1;
        else
            output[ind] = 0;
    }
}

__global__ void bool_matrix_broadcast_kernel(const float *input_A,
                                             const float *input_B,
                                             float *output, int cond, int ncol,
                                             size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    int row = ind / ncol;
    int col = ind % ncol;
    if (cond == 0) {
        if (abs(input_A[col] - input_B[row]) < 1e-6)
            output[ind] = 1;
        else
            output[ind] = 0;
    } else if (cond == 1) {
        if (input_A[col] - input_B[row] < -1e-6)
            output[ind] = 1;
        else
            output[ind] = 0;
    } else {
        if (input_A[col] - input_B[row] > 1e-6)
            output[ind] = 1;
        else
            output[ind] = 0;
    }
}

int DLGpuBoolMatrix(const DLArrayHandle input_A, const DLArrayHandle input_B,
                    DLArrayHandle output, int cond,
                    DLStreamHandle stream_handle = NULL) {
    size_t size = 1;
    int ncol = 0;
    for (index_t i = 0; i < output->ndim; i++) {
        size *= output->shape[i];
    }
    if (input_A->ndim != input_B->ndim) {
        assert(input_A->ndim == 1);
        assert(input_B->ndim == 2);
        assert(output->ndim == 2);
        ncol = output->shape[1];
    }

    dim3 blocks;
    dim3 threads;
    float *input_A_data = (float *)input_A->data;
    float *input_B_data = (float *)input_B->data;
    float *output_data = (float *)output->data;
    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }
    if (stream_handle)
        if (input_A->ndim == input_B->ndim)
            bool_matrix_kernel<<<blocks, threads, 0,
                                 *(cudaStream_t *)stream_handle->handle>>>(
                input_A_data, input_B_data, output_data, cond, size);
        else
            bool_matrix_broadcast_kernel<<<
                blocks, threads, 0, *(cudaStream_t *)stream_handle->handle>>>(
                input_A_data, input_B_data, output_data, cond, ncol, size);
    else if (input_A->ndim == input_B->ndim)
        bool_matrix_kernel<<<blocks, threads>>>(input_A_data, input_B_data,
                                                output_data, cond, size);
    else
        bool_matrix_broadcast_kernel<<<blocks, threads>>>(
            input_A_data, input_B_data, output_data, cond, ncol, size);
    return 0;
}