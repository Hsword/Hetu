#include "gpu_runtime.h"
#include <math.h>

__global__ void flip_kernel(const float *input, float *output, int64_t *shape,
                            int64_t *stride, int *flip_dim, int nflip, int ndim,
                            size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    int tmp = ind;
    int cnt = 0;
    int id;
    int input_ind = 0;
    for (int i = 0; i < ndim; i++) {
        id = tmp / stride[i];
        if (i == flip_dim[cnt]) {
            input_ind += (shape[i] - 1 - id) * stride[i];
            ++cnt;
        } else
            input_ind += id * stride[i];
        tmp %= stride[i];
    }
    output[ind] = input[input_ind];
}

int DLGpuFlip(const DLArrayHandle input, DLArrayHandle output, int *dims,
              int nflip, DLStreamHandle stream_handle = NULL) {
    size_t size = 1;
    int ndim = input->ndim;
    cudaStream_t cu_stream = static_cast<cudaStream_t>(
        stream_handle ? *(cudaStream_t *)(stream_handle->handle) : NULL);
    for (index_t i = 0; i < ndim; i++) {
        size *= input->shape[i];
    }

    int dev_id = (input->ctx).device_id;
    if (is_chunk_init(dev_id) == false) {
        chunk_init(dev_id);
    }

    int *flip_dim = (int *)find_chunk(ndim * sizeof(int), dev_id);
    int64_t *shape = (int64_t *)find_chunk(ndim * sizeof(int64_t), dev_id);
    int64_t *stride = (int64_t *)find_chunk(ndim * sizeof(int64_t), dev_id);

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
    if (stream_handle) {
        CUDA_CALL(cudaMemcpyAsync(flip_dim, (void *)dims, ndim * sizeof(int),
                                  cudaMemcpyHostToDevice, cu_stream));
        CUDA_CALL(cudaMemcpyAsync(shape, (void *)(input->shape),
                                  ndim * sizeof(int64_t),
                                  cudaMemcpyHostToDevice, cu_stream));
        CUDA_CALL(cudaMemcpyAsync(stride, (void *)(input->stride),
                                  ndim * sizeof(int64_t),
                                  cudaMemcpyHostToDevice, cu_stream));
        flip_kernel<<<blocks, threads, 0,
                      *(cudaStream_t *)stream_handle->handle>>>(
            input_data, output_data, shape, stride, flip_dim, nflip, ndim,
            size);
    } else {
        CUDA_CALL(cudaMemcpyAsync(flip_dim, (void *)dims, ndim * sizeof(int),
                                  cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpyAsync(shape, (void *)(input->shape),
                                  ndim * sizeof(int64_t),
                                  cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpyAsync(stride, (void *)(input->stride),
                                  ndim * sizeof(int64_t),
                                  cudaMemcpyHostToDevice));
        flip_kernel<<<blocks, threads>>>(input_data, output_data, shape, stride,
                                         flip_dim, nflip, ndim, size);
    }
    return 0;
}
