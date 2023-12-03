#include "gpu_runtime.h"

__global__ void transpose_kernel(float *odata, const float *idata,
                                 const uint *buf, const uint ndims,
                                 size_t size) {
    size_t o_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (o_idx >= size)
        return;
    const uint *in_strides = buf;
    const uint *out_strides = buf + ndims;
    const uint *perm = buf + ndims * 2;

    uint i_idx = 0;
    uint t = o_idx;
    for (int i = 0; i < ndims; ++i) {
        const uint ratio = t / out_strides[i];
        t -= ratio * out_strides[i];
        i_idx += ratio * in_strides[perm[i]];
    }
    odata[o_idx] = idata[i_idx];
}

int DLGpuTranspose(const DLArrayHandle input, DLArrayHandle output, int *perm,
                   DLStreamHandle stream_handle = NULL) {
    uint ndim = uint(input->ndim);
    uint ndim_ = uint(output->ndim);
    assert(ndim == ndim_);

    int64_t *in_dims = input->shape;
    int64_t *out_dims = output->shape;
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;

    uint *buf = (uint *)malloc(3 * ndim * sizeof(uint));
    uint *gpu_buf = NULL;

    uint in_stride = 1;
    uint out_stride = 1;
    for (int i = ndim - 1; i >= 0; --i) {
        buf[i] = uint(in_stride);
        buf[ndim + i] = uint(out_stride);
        buf[ndim * 2 + i] = uint(perm[i]);
        in_stride *= uint(in_dims[i]);
        out_stride *= uint(out_dims[i]);
    }

    assert(in_stride == out_stride);
    size_t size = in_stride;

    int dev_id = (input->ctx).device_id;
    if (is_chunk_init(dev_id) == false) {
        chunk_init(dev_id);
    }
    size_t buf_size = 3 * ndim * sizeof(uint);
    gpu_buf = (uint *)find_chunk(buf_size, dev_id);

    dim3 blocks;
    dim3 threads;
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
        CUDA_CALL(cudaMemcpyAsync(gpu_buf, (void *)buf, buf_size,
                                  cudaMemcpyHostToDevice, cu_stream));
        transpose_kernel<<<blocks, threads, 0, cu_stream>>>(
            output_data, input_data, gpu_buf, ndim, size);
    } else {
        CUDA_CALL(
            cudaMemcpy(gpu_buf, (void *)buf, buf_size, cudaMemcpyHostToDevice));
        transpose_kernel<<<blocks, threads>>>(output_data, input_data, gpu_buf,
                                              ndim, size);
    }

    del_chunk(gpu_buf, dev_id);
    free(buf);
    return 0;
}

int DLGpuTransposeSimple(const DLArrayHandle input, DLArrayHandle output,
                         const DLArrayHandle gpu_buffer,
                         DLStreamHandle stream_handle = NULL) {
    const float *input_data = (const float *)input->data;
    float *output_data = (float *)output->data;
    const uint *gpu_buf = (const uint *)gpu_buffer->data;
    const uint ndim = output->ndim;

    size_t size = 1;
    for (uint i = 0; i < ndim; ++i) {
        size *= output->shape[i];
    }

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
        cudaStream_t cu_stream = (*(cudaStream_t *)(stream_handle->handle));
        transpose_kernel<<<blocks, threads, 0, cu_stream>>>(
            output_data, input_data, gpu_buf, ndim, size);
    } else {
        transpose_kernel<<<blocks, threads>>>(output_data, input_data, gpu_buf,
                                              ndim, size);
    }

    return 0;
}