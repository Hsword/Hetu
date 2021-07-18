#include "gpu_runtime.h"

__global__ void slice_kernel(float *out_arr, const float *in_arr,
                             const int64_t *o_shape, const int64_t *i_shape,
                             const int64_t *begin_pos, size_t ndim,
                             size_t size) {
    size_t o_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (o_index >= size)
        return;

    size_t tmp_index = o_index;
    size_t i_index = 0;
    int64_t i_mat = 1;

    for (int i = ndim - 1; i >= 0; --i) {
        int64_t offset = begin_pos[i] + tmp_index % o_shape[i];
        tmp_index /= o_shape[i];
        i_index += offset * i_mat;
        i_mat *= i_shape[i];
    }
    out_arr[o_index] = in_arr[i_index];
}

int DLGpuSlice(const DLArrayHandle in_arr, DLArrayHandle out_arr,
               int64_t *begin_pos, DLStreamHandle stream_handle = NULL) {
    assert(in_arr->ndim == out_arr->ndim);
    size_t ndim = in_arr->ndim;
    size_t o_size = 1;
    for (int i = 0; i < ndim; ++i) {
        assert(begin_pos[i] >= 0);
        assert(begin_pos[i] + out_arr->shape[i] <= in_arr->shape[i]);
        o_size *= out_arr->shape[i];
    }
    const float *i_data = (const float *)in_arr->data;
    float *o_data = (float *)out_arr->data;
    int dev_id = (in_arr->ctx).device_id;
    if (is_chunk_init(dev_id) == false) {
        chunk_init(dev_id);
    }
    size_t alloc_size = ndim * sizeof(int64_t);
    void *pos = find_chunk(alloc_size, dev_id);
    void *i_shape = find_chunk(alloc_size, dev_id);
    void *o_shape = find_chunk(alloc_size, dev_id);

    dim3 blocks;
    dim3 threads;
    if (o_size <= 1024) {
        threads.x = o_size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (o_size + 1023) / 1024;
    }

    cudaStream_t cu_stream = static_cast<cudaStream_t>(
        stream_handle ? *(cudaStream_t *)(stream_handle->handle) : NULL);

    if (cu_stream != NULL) {
        CUDA_CALL(cudaMemcpyAsync(pos, (void *)begin_pos, alloc_size,
                                  cudaMemcpyHostToDevice, cu_stream));
        CUDA_CALL(cudaMemcpyAsync(i_shape, (void *)in_arr->shape, alloc_size,
                                  cudaMemcpyHostToDevice, cu_stream));
        CUDA_CALL(cudaMemcpyAsync(o_shape, (void *)out_arr->shape, alloc_size,
                                  cudaMemcpyHostToDevice, cu_stream));
        slice_kernel<<<blocks, threads, 0, cu_stream>>>(
            o_data, i_data, (const int64_t *)o_shape, (const int64_t *)i_shape,
            (const int64_t *)pos, ndim, o_size);

    } else {
        CUDA_CALL(cudaMemcpy(pos, (void *)begin_pos, alloc_size,
                             cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(i_shape, (void *)in_arr->shape, alloc_size,
                             cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(o_shape, (void *)out_arr->shape, alloc_size,
                             cudaMemcpyHostToDevice));
        slice_kernel<<<blocks, threads>>>(
            o_data, i_data, (const int64_t *)o_shape, (const int64_t *)i_shape,
            (const int64_t *)pos, ndim, o_size);
    }

    del_chunk(o_shape, dev_id);
    del_chunk(i_shape, dev_id);
    del_chunk(pos, dev_id);
    return 0;
}

__global__ void slice_gradient_kernel(float *out_arr, const float *in_arr,
                                      const int64_t *o_shape,
                                      const int64_t *i_shape,
                                      const int64_t *begin_pos, size_t ndim,
                                      size_t size) {
    size_t o_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (o_index >= size)
        return;

    out_arr[o_index] = 0;

    size_t tmp_index = o_index;
    size_t i_index = 0;
    int64_t i_mat = 1;

    for (int i = ndim - 1; i >= 0; --i) {
        int64_t offset = tmp_index % o_shape[i];
        if (offset < begin_pos[i] || offset >= begin_pos[i] + i_shape[i])
            return;
        tmp_index /= o_shape[i];
        i_index += (offset - begin_pos[i]) * i_mat;
        i_mat *= i_shape[i];
    }
    out_arr[o_index] = in_arr[i_index];
}

int DLGpuSliceGradient(const DLArrayHandle in_arr, DLArrayHandle out_arr,
                       int64_t *begin_pos,
                       DLStreamHandle stream_handle = NULL) {
    assert(in_arr->ndim == out_arr->ndim);
    size_t ndim = in_arr->ndim;
    size_t o_size = 1;
    for (int i = 0; i < ndim; ++i) {
        assert(begin_pos[i] >= 0);
        assert(begin_pos[i] + in_arr->shape[i] <= out_arr->shape[i]);
        o_size *= out_arr->shape[i];
    }
    const float *i_data = (const float *)in_arr->data;
    float *o_data = (float *)out_arr->data;
    int dev_id = (in_arr->ctx).device_id;
    if (is_chunk_init(dev_id) == false) {
        chunk_init(dev_id);
    }
    size_t alloc_size = ndim * sizeof(int64_t);
    void *pos = find_chunk(alloc_size, dev_id);
    void *i_shape = find_chunk(alloc_size, dev_id);
    void *o_shape = find_chunk(alloc_size, dev_id);

    dim3 blocks;
    dim3 threads;
    if (o_size <= 1024) {
        threads.x = o_size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (o_size + 1023) / 1024;
    }

    cudaStream_t cu_stream = static_cast<cudaStream_t>(
        stream_handle ? *(cudaStream_t *)(stream_handle->handle) : NULL);

    if (cu_stream != NULL) {
        CUDA_CALL(cudaMemcpyAsync(pos, (void *)begin_pos, alloc_size,
                                  cudaMemcpyHostToDevice, cu_stream));
        CUDA_CALL(cudaMemcpyAsync(i_shape, (void *)in_arr->shape, alloc_size,
                                  cudaMemcpyHostToDevice, cu_stream));
        CUDA_CALL(cudaMemcpyAsync(o_shape, (void *)out_arr->shape, alloc_size,
                                  cudaMemcpyHostToDevice, cu_stream));
        slice_gradient_kernel<<<blocks, threads, 0, cu_stream>>>(
            o_data, i_data, (const int64_t *)o_shape, (const int64_t *)i_shape,
            (const int64_t *)pos, ndim, o_size);
    } else {
        CUDA_CALL(cudaMemcpy(pos, (void *)begin_pos, alloc_size,
                             cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(i_shape, (void *)in_arr->shape, alloc_size,
                             cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(o_shape, (void *)out_arr->shape, alloc_size,
                             cudaMemcpyHostToDevice));
        slice_gradient_kernel<<<blocks, threads>>>(
            o_data, i_data, (const int64_t *)o_shape, (const int64_t *)i_shape,
            (const int64_t *)pos, ndim, o_size);
    }

    del_chunk(o_shape, dev_id);
    del_chunk(i_shape, dev_id);
    del_chunk(pos, dev_id);
    return 0;
}

/* below is the simple version of slice and slicegradient */

__global__ void slice_kernel_simple(float *out_arr, const float *in_arr,
                                    const uint *gpu_buf, size_t ndim,
                                    size_t size) {
    size_t o_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (o_index >= size)
        return;

    const uint *begin_pos = gpu_buf;
    const uint *i_shape = begin_pos + ndim;
    const uint *o_shape = i_shape + ndim;

    size_t tmp_index = o_index;
    size_t i_index = 0;
    uint i_mat = 1;

    for (int i = ndim - 1; i >= 0; --i) {
        uint offset = begin_pos[i] + tmp_index % o_shape[i];
        tmp_index /= o_shape[i];
        i_index += offset * i_mat;
        i_mat *= i_shape[i];
    }
    out_arr[o_index] = in_arr[i_index];
}

int DLGpuSliceSimple(const DLArrayHandle in_arr, DLArrayHandle out_arr,
                     const DLArrayHandle gpu_buf,
                     DLStreamHandle stream_handle = NULL) {
    assert(in_arr->ndim == out_arr->ndim);
    size_t ndim = in_arr->ndim;
    size_t o_size = 1;
    for (int i = 0; i < ndim; ++i) {
        o_size *= out_arr->shape[i];
    }

    dim3 blocks;
    dim3 threads;
    if (o_size <= 1024) {
        threads.x = o_size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (o_size + 1023) / 1024;
    }

    if (stream_handle) {
        cudaStream_t cu_stream = *(cudaStream_t *)(stream_handle->handle);
        slice_kernel_simple<<<blocks, threads, 0, cu_stream>>>(
            (float *)(out_arr->data), (const float *)(in_arr->data),
            (const uint *)(gpu_buf->data), ndim, o_size);

    } else {
        slice_kernel_simple<<<blocks, threads>>>(
            (float *)(out_arr->data), (const float *)(in_arr->data),
            (const uint *)(gpu_buf->data), ndim, o_size);
    }
    return 0;
}

__global__ void slice_gradient_kernel_simple(float *out_arr,
                                             const float *in_arr,
                                             const uint *gpu_buf, size_t ndim,
                                             size_t size) {
    size_t o_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (o_index >= size)
        return;

    const uint *begin_pos = gpu_buf;
    const uint *i_shape = begin_pos + ndim;
    const uint *o_shape = i_shape + ndim;

    out_arr[o_index] = 0;

    size_t tmp_index = o_index;
    size_t i_index = 0;
    int64_t i_mat = 1;

    for (int i = ndim - 1; i >= 0; --i) {
        int64_t offset = tmp_index % o_shape[i];
        if (offset < begin_pos[i] || offset >= begin_pos[i] + i_shape[i])
            return;
        tmp_index /= o_shape[i];
        i_index += (offset - begin_pos[i]) * i_mat;
        i_mat *= i_shape[i];
    }
    out_arr[o_index] = in_arr[i_index];
}

int DLGpuSliceGradientSimple(const DLArrayHandle in_arr, DLArrayHandle out_arr,
                             const DLArrayHandle gpu_buf,
                             DLStreamHandle stream_handle = NULL) {
    assert(in_arr->ndim == out_arr->ndim);
    size_t ndim = in_arr->ndim;
    size_t o_size = 1;
    for (int i = 0; i < ndim; ++i) {
        o_size *= out_arr->shape[i];
    }

    dim3 blocks;
    dim3 threads;
    if (o_size <= 1024) {
        threads.x = o_size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (o_size + 1023) / 1024;
    }

    if (stream_handle) {
        cudaStream_t cu_stream = *(cudaStream_t *)(stream_handle->handle);
        slice_gradient_kernel_simple<<<blocks, threads, 0, cu_stream>>>(
            (float *)(out_arr->data), (const float *)(in_arr->data),
            (const uint *)(gpu_buf->data), ndim, o_size);
    } else {
        slice_gradient_kernel_simple<<<blocks, threads>>>(
            (float *)(out_arr->data), (const float *)(in_arr->data),
            (const uint *)(gpu_buf->data), ndim, o_size);
    }
    return 0;
}
