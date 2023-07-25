#include "gpu_runtime.h"
#include <curand_kernel.h>

__global__ void as_stride_kernel(const float *input, float *output,
                                 const int *in_stride,
                                 const int64_t *out_stride, int ndim,
                                 size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    int tmp = ind;
    int input_ind = 0;
    int id;

    for (int i = 0; i < ndim; i++) {
        id = tmp / out_stride[i];
        input_ind += id * in_stride[i];
        tmp = tmp % out_stride[i];
    }
    output[ind] = input[input_ind];
}

int DLGpuAsStrided(const DLArrayHandle input, DLArrayHandle output, int *stride,
                   DLStreamHandle stream_handle = NULL) {
    size_t size = 1;
    int ndim = output->ndim;
    cudaStream_t cu_stream = static_cast<cudaStream_t>(
        stream_handle ? *(cudaStream_t *)(stream_handle->handle) : NULL);

    for (index_t i = 0; i < ndim; i++) {
        size *= output->shape[i];
    }

    int *in_stride = NULL;
    int64_t *out_stride = NULL;

    int dev_id = (input->ctx).device_id;
    if (is_chunk_init(dev_id) == false) {
        chunk_init(dev_id);
    }
    size_t in_stride_size = ndim * sizeof(int);
    size_t out_stride_size = ndim * sizeof(int64_t);

    in_stride = (int *)find_chunk(in_stride_size, dev_id);
    out_stride = (int64_t *)find_chunk(out_stride_size, dev_id);

    const float *input_data = (const float *)input->data;
    float *output_data = (float *)output->data;

    dim3 blocks;
    dim3 threads;
    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }
    if (stream_handle) {
        CUDA_CALL(cudaMemcpyAsync(in_stride, (void *)stride, in_stride_size,
                                  cudaMemcpyHostToDevice, cu_stream));
        CUDA_CALL(cudaMemcpyAsync(out_stride, (void *)(output->stride),
                                  out_stride_size, cudaMemcpyHostToDevice,
                                  cu_stream));
        as_stride_kernel<<<blocks, threads, 0, cu_stream>>>(
            input_data, output_data, (const int *)in_stride,
            (const int64_t *)out_stride, ndim, size);
    } else {
        CUDA_CALL(cudaMemcpyAsync(in_stride, (void *)stride, in_stride_size,
                                  cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpyAsync(out_stride, (void *)(output->stride),
                                  out_stride_size, cudaMemcpyHostToDevice));
        as_stride_kernel<<<blocks, threads>>>(
            input_data, output_data, (const int *)in_stride,
            (const int64_t *)out_stride, ndim, size);
    }
    return 0;
}

__global__ void as_stride_gradient_kernel(const float *grad, float *output,
                                          const int *in_stride,
                                          const int64_t *out_stride, int ndim,
                                          size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    int tmp = ind;
    int input_ind = 0;
    int id;
    for (int i = 0; i < ndim; i++) {
        id = tmp / out_stride[i];
        input_ind += id * in_stride[i];
        tmp = tmp % out_stride[i];
    }
    atomicAdd(&output[input_ind], grad[ind]);
}

int DLGpuAsStridedGradient(const DLArrayHandle grad, DLArrayHandle output,
                           int *stride, DLStreamHandle stream_handle = NULL) {
    size_t size = 1;
    int ndim = grad->ndim;
    cudaStream_t cu_stream = static_cast<cudaStream_t>(
        stream_handle ? *(cudaStream_t *)(stream_handle->handle) : NULL);

    for (index_t i = 0; i < ndim; i++) {
        size *= grad->shape[i];
    }

    int *in_stride = NULL;
    int64_t *out_stride = NULL;

    int dev_id = (grad->ctx).device_id;
    if (is_chunk_init(dev_id) == false) {
        chunk_init(dev_id);
    }
    size_t in_stride_size = ndim * sizeof(int);
    size_t out_stride_size = ndim * sizeof(int64_t);

    in_stride = (int *)find_chunk(in_stride_size, dev_id);
    out_stride = (int64_t *)find_chunk(out_stride_size, dev_id);

    const float *grad_data = (const float *)grad->data;
    float *output_data = (float *)output->data;

    dim3 blocks;
    dim3 threads;
    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }
    if (stream_handle) {
        CUDA_CALL(cudaMemcpyAsync(in_stride, (void *)stride, in_stride_size,
                                  cudaMemcpyHostToDevice, cu_stream));
        CUDA_CALL(cudaMemcpyAsync(out_stride, (void *)(grad->stride),
                                  out_stride_size, cudaMemcpyHostToDevice,
                                  cu_stream));
        as_stride_gradient_kernel<<<blocks, threads, 0, cu_stream>>>(
            grad_data, output_data, (const int *)in_stride,
            (const int64_t *)out_stride, ndim, size);
    } else {
        CUDA_CALL(cudaMemcpyAsync(in_stride, (void *)stride, in_stride_size,
                                  cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpyAsync(out_stride, (void *)(grad->stride),
                                  out_stride_size, cudaMemcpyHostToDevice));
        as_stride_gradient_kernel<<<blocks, threads>>>(
            grad_data, output_data, (const int *)in_stride,
            (const int64_t *)out_stride, ndim, size);
    }
    return 0;
}