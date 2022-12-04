#include "gpu_runtime.h"

__global__ void robe_lookup_kernel(const float *input, const int *ids, const int *x,
                                        float *output, size_t size,
                                        size_t length, int Bg, int Cg, int Dg, size_t roarsz) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size)
        return;
    int id = ids[index];
    float *output_ptr = output + length * index;
    if (id < 0 || id >= roarsz) {
        for (int i = 0; i < length; i++)
            output_ptr[i] = 0;
    } else {
        //const float *input_ptr = input + id;
        int X = x[index];
        for (int i = 0; i < length; i++){
            int sgn = (((1ll * X * Bg + 1ll * i * Cg + Dg)%998244353+998244353)%998244353 %2)*2-1;
            output_ptr[i] = input[i+id<roarsz?(i+id):(i+id-roarsz)] * sgn;
        }
    }
}

int DLGpuRobeLookUp(const DLArrayHandle input, const DLArrayHandle ids, const DLArrayHandle x,
                         DLArrayHandle output, int len, int Bg, int Cg, int Dg,
                         DLStreamHandle stream_handle = NULL) {
    assert(input->ndim == 1);
    size_t size = 1;
    for (int i = 0; i < output->ndim; i++) {
        if (i < output->ndim - 1) {
            assert(ids->shape[i] == output->shape[i]);
        } else if (i == output->ndim - 1) {
            assert(len == output->shape[i]);
        }
    }
    for (int i = 0; i < ids->ndim; i++) {
        size = size * ids->shape[i];
    }
    size_t roarsz = input->shape[0];
    size_t length = (size_t)(len);
    dim3 blocks;
    dim3 threads;
    float *output_data = (float *)output->data;
    const float *input_data = (const float *)input->data;
    const int *id_list = (const int *)ids->data;
    const int *x_list = (const int *)x->data;
    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }
    if (stream_handle)
        robe_lookup_kernel<<<blocks, threads, 0,
                                  *(cudaStream_t *)stream_handle->handle>>>(
            input_data, id_list, x_list, output_data, size, length, roarsz, Bg, Cg, Dg);
    else
        robe_lookup_kernel<<<blocks, threads>>>(
            input_data, id_list, x_list, output_data, size, length, roarsz, Bg, Cg, Dg);
    return 0;
}

__global__ void array_set_zero_kernel1(float *output, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    output[ind] = 0;
}


__global__ void robe_lookup_gradient_kernel(const float *output_grad_data,
                                                 const int *ids,
                                                 float *input_grad_data,
                                                 size_t size, size_t length) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size)
        return;
    int id = ids[index];
    const float *output_grad_ptr = output_grad_data + length * index;
    float *input_grad_ptr = input_grad_data + length * id;
    for (int i = 0; i < length; i++)
        atomicAdd(input_grad_ptr + i, *(output_grad_ptr + i));
}

int DLGpuRobeLookUp_Gradient(const DLArrayHandle output_grad,
                                  const DLArrayHandle ids,
                                  DLArrayHandle input_grad,
                                  DLStreamHandle stream_handle = NULL) {
    assert(input_grad->ndim == 2);
    size_t size = 1;
    for (int i = 0; i < output_grad->ndim; i++) {
        if (i < output_grad->ndim - 1) {
            assert(ids->shape[i] == output_grad->shape[i]);
        } else if (i == output_grad->ndim - 1) {
            assert(input_grad->shape[1] == output_grad->shape[i]);
        }
    }
    for (int i = 0; i < ids->ndim; i++) {
        size = size * ids->shape[i];
    }
    size_t length = input_grad->shape[1];
    dim3 blocks;
    dim3 threads;
    const float *output_grad_data = (const float *)output_grad->data;
    float *input_grad_data = (float *)input_grad->data;
    const int *id_list = (const int *)ids->data;

    size_t input_grad_size = 1;
    for (int i = 0; i < input_grad->ndim; i++) {
        input_grad_size *= input_grad->shape[i];
    }
    if (input_grad_size <= 1024) {
        threads.x = input_grad_size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (input_grad_size + 1023) / 1024;
    }
    if (stream_handle)
        array_set_zero_kernel1<<<blocks, threads, 0,
                                *(cudaStream_t *)stream_handle->handle>>>(
            input_grad_data, input_grad_size);
    else
        array_set_zero_kernel1<<<blocks, threads>>>(input_grad_data,
                                                   input_grad_size);

    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }
    if (stream_handle)
        robe_lookup_gradient_kernel<<<
            blocks, threads, 0, *(cudaStream_t *)stream_handle->handle>>>(
            output_grad_data, id_list, input_grad_data, size, length);
    else
        robe_lookup_gradient_kernel<<<blocks, threads>>>(
            output_grad_data, id_list, input_grad_data, size, length);
    return 0;
}