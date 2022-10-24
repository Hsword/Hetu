#include "gpu_runtime.h"

__global__ void repeat_kernel(const float *input, float *output, size_t size,
                              int64_t *stride_in, int64_t *stride_out,
                              int64_t *dim, int ndim) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    size_t tmp = ind;
    if (ind >= size)
        return;
    int index = 0;
    for (int i = 0; i < ndim; i++) {
        int tmp_index = ind / stride_out[i];
        index += (tmp_index % dim[i]) * stride_in[i];
        ind -= tmp_index * stride_out[i];
    }
    output[tmp] = input[index];
}

int DLGpuRepeat(const DLArrayHandle input, DLArrayHandle output,
                DLStreamHandle stream_handle = NULL) {
    size_t size = 1;
    int ndim = output->ndim;
    for (index_t i = 0; i < ndim; i++) {
        size *= output->shape[i];
    }
    int64_t *stride_tmp = new int64_t[ndim];
    int64_t *shape_tmp = new int64_t[ndim];
    for (int i = 0; i < ndim; i++) {
        if (i < (ndim - input->ndim)) {
            stride_tmp[i] = input->stride[0];
            shape_tmp[i] = 1;
        } else {
            stride_tmp[i] = input->stride[i - (ndim - input->ndim)];
            shape_tmp[i] = input->shape[i - (ndim - input->ndim)];
        }
    }
    int64_t *stride_in = NULL;
    int64_t *stride_out = NULL;
    int64_t *dim = NULL;
    int dev_id = (input->ctx).device_id;
    if (is_chunk_init(dev_id) == false) {
        chunk_init(dev_id);
    }
    size_t buf_size = 3 * ndim * sizeof(int64_t);
    stride_in = (int64_t *)find_chunk(buf_size, dev_id);
    stride_out = (int64_t *)find_chunk(buf_size, dev_id);
    dim = (int64_t *)find_chunk(buf_size, dev_id);

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
    cudaStream_t cu_stream = static_cast<cudaStream_t>(
        stream_handle ? *(cudaStream_t *)(stream_handle->handle) : NULL);
    if (cu_stream) {
        CUDA_CALL(cudaMemcpyAsync(stride_in, (void *)stride_tmp, buf_size,
                                  cudaMemcpyHostToDevice, cu_stream));
        CUDA_CALL(cudaMemcpyAsync(stride_out, (void *)output->stride, buf_size,
                                  cudaMemcpyHostToDevice, cu_stream));
        CUDA_CALL(cudaMemcpyAsync(dim, (void *)shape_tmp, buf_size,
                                  cudaMemcpyHostToDevice, cu_stream));
        repeat_kernel<<<blocks, threads, 0, cu_stream>>>(
            input_data, output_data, size, stride_in, stride_out, dim, ndim);
    } else {
        CUDA_CALL(cudaMemcpyAsync(stride_in, (void *)stride_tmp, buf_size,
                                  cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpyAsync(stride_out, (void *)output->stride, buf_size,
                                  cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpyAsync(dim, (void *)shape_tmp, buf_size,
                                  cudaMemcpyHostToDevice));
        repeat_kernel<<<blocks, threads>>>(input_data, output_data, size,
                                           stride_in, stride_out, dim, ndim);
    }
    return 0;
}

__global__ void repeat_gradient_kernel(const float *input, float *output,
                                       size_t size, int64_t *stride_in,
                                       int64_t *stride_out, int64_t *dim,
                                       int ndim) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    size_t tmp = ind;
    if (ind >= size)
        return;
    int index = 0;
    for (int i = 0; i < ndim; i++) {
        int tmp_index = ind / stride_in[i];
        index += (tmp_index % dim[i]) * stride_out[i];
        ind -= tmp_index * stride_in[i];
    }
    atomicAdd(&output[index], input[tmp]);
}

int DLGpuRepeatGradient(const DLArrayHandle input, DLArrayHandle output,
                        DLStreamHandle stream_handle = NULL) {
    size_t size = 1;
    int ndim = input->ndim;
    for (index_t i = 0; i < ndim; i++) {
        size *= input->shape[i];
    }
    int64_t *stride_tmp = new int64_t[ndim];
    int64_t *shape_tmp = new int64_t[ndim];
    for (int i = 0; i < ndim; i++) {
        if (i < (ndim - output->ndim)) {
            stride_tmp[i] = output->stride[0];
            shape_tmp[i] = 1;
        } else {
            stride_tmp[i] = output->stride[i - (ndim - output->ndim)];
            shape_tmp[i] = output->shape[i - (ndim - output->ndim)];
        }
    }
    int64_t *stride_in = NULL;
    int64_t *stride_out = NULL;
    int64_t *dim = NULL;
    int dev_id = (input->ctx).device_id;
    if (is_chunk_init(dev_id) == false) {
        chunk_init(dev_id);
    }
    size_t buf_size = 3 * ndim * sizeof(int64_t);
    stride_in = (int64_t *)find_chunk(buf_size, dev_id);
    stride_out = (int64_t *)find_chunk(buf_size, dev_id);
    dim = (int64_t *)find_chunk(buf_size, dev_id);

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
    cudaStream_t cu_stream = static_cast<cudaStream_t>(
        stream_handle ? *(cudaStream_t *)(stream_handle->handle) : NULL);
    if (cu_stream) {
        CUDA_CALL(cudaMemcpyAsync(stride_in, (void *)input->stride, buf_size,
                                  cudaMemcpyHostToDevice, cu_stream));
        CUDA_CALL(cudaMemcpyAsync(stride_out, (void *)stride_tmp, buf_size,
                                  cudaMemcpyHostToDevice, cu_stream));
        CUDA_CALL(cudaMemcpyAsync(dim, (void *)shape_tmp, buf_size,
                                  cudaMemcpyHostToDevice, cu_stream));
        repeat_gradient_kernel<<<blocks, threads, 0, cu_stream>>>(
            input_data, output_data, size, stride_in, stride_out, dim, ndim);
    } else {
        CUDA_CALL(cudaMemcpyAsync(stride_in, (void *)input->stride, buf_size,
                                  cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpyAsync(stride_out, (void *)stride_tmp, buf_size,
                                  cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpyAsync(dim, (void *)shape_tmp, buf_size,
                                  cudaMemcpyHostToDevice));
        repeat_gradient_kernel<<<blocks, threads>>>(
            input_data, output_data, size, stride_in, stride_out, dim, ndim);
    }
    return 0;
}
