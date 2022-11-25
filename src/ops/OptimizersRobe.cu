#include "gpu_runtime.h"


__global__ void sgd_robe_update(const float *grad_data,
                                  const int *indices_data, float *param_data,
                                  size_t size, size_t length, float lr, int roarsz) {
    size_t thread_ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_ind >= size)
        return;
    size_t index = thread_ind / length;
    size_t offset = thread_ind % length;
    int id = indices_data[index];
    if (id < 0)
        return;
    const float cur_grad = grad_data[thread_ind];
    atomicAdd(param_data + (id + offset<roarsz?id+offset:id+offset-roarsz), -lr * cur_grad);
//    param_data[(id + offset<roarsz?id+offset:id+offset-roarsz)] -= lr * cur_grad;
}

int SGDOptimizerRobeUpdate(DLArrayHandle param,
                             const DLArrayHandle grad_indices,
                             const DLArrayHandle grad_values, float lr,
                             DLStreamHandle stream_handle = NULL) {
    size_t size = ArrSize(grad_values);
    size_t roarsz = ArrSize(param);
    
    size_t length = (grad_values->shape[(grad_values->ndim) - 1]);

    const float *grad_data = (const float *)grad_values->data;
    float *param_data = (float *)param->data;
    const int *indices_data = (const int *)grad_indices->data;

    dim3 blocks;
    dim3 threads;
    ThreadBlock1D(threads, blocks, size);

    if (stream_handle)
        sgd_robe_update<<<blocks, threads, 0,
                            *(cudaStream_t *)stream_handle->handle>>>(
            grad_data, indices_data, param_data, size, length, lr, roarsz);
    else
        sgd_robe_update<<<blocks, threads>>>(grad_data, indices_data,
                                               param_data, size, length, lr, roarsz);
    return 0;
}
