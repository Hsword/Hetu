#include "gpu_runtime.h"

__global__ void array_lazy_callback_kernel(const float *from, float *to,
                                           index_t *from_stride,
                                           index_t *to_stride, size_t size,
                                           size_t ndim) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    size_t to_index = 0;
    size_t from_index = ind;
    for (index_t i = 0; i < ndim; i++) {
        to_index = to_index + from_index / to_stride[i] * from_stride[i];
        from_index = from_index % to_stride[i];
    }
    to[ind] = from[to_index];
}

int DLGpuArrayLazyCallback(const DLArrayHandle from, DLArrayHandle to,
                           DLStreamHandle stream_handle = NULL) {
    int dev_id = (from->ctx).device_id;
    cudaSetDevice(dev_id);
    index_t size = 1;
    index_t ndim = to->ndim;
    for (index_t i = 0; i < ndim; i++) {
        size *= to->shape[i];
    }

    if (is_chunk_init(dev_id) == false) {
        chunk_init(dev_id);
    }

    index_t *from_stride =
        (index_t *)find_chunk(ndim * sizeof(index_t), dev_id);
    index_t *to_stride = (index_t *)find_chunk(ndim * sizeof(index_t), dev_id);

    dim3 blocks;
    dim3 threads;
    float *to_data = (float *)to->data;
    const float *from_data = (const float *)from->data;
    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }

    cudaStream_t cu_stream = static_cast<cudaStream_t>(
        stream_handle ? *(cudaStream_t *)(stream_handle->handle) : NULL);
    if (cu_stream != NULL) {
        CUDA_CALL(cudaMemcpyAsync(from_stride, from->stride,
                                  ndim * sizeof(index_t),
                                  cudaMemcpyHostToDevice, cu_stream));
        CUDA_CALL(cudaMemcpyAsync(to_stride, to->stride, ndim * sizeof(index_t),
                                  cudaMemcpyHostToDevice, cu_stream));
        array_lazy_callback_kernel<<<blocks, threads, 0, cu_stream>>>(
            from_data, to_data, from_stride, to_stride, size, ndim);
    } else {
        CUDA_CALL(cudaMemcpy(from_stride, from->stride, ndim * sizeof(index_t),
                             cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(to_stride, to->stride, ndim * sizeof(index_t),
                             cudaMemcpyHostToDevice));
        array_lazy_callback_kernel<<<blocks, threads>>>(
            from_data, to_data, from_stride, to_stride, size, ndim);
    }

    del_chunk(from_stride, dev_id);
    del_chunk(to_stride, dev_id);
    return 0;
}