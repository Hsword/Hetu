#include "gpu_runtime.h"

__global__ void slice_assign(const float *input, float *output, float val,
                             int *begin_pos, int *end_pos, int64_t *stride,
                             size_t size, size_t ndim) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    size_t tmp = ind;
    if (ind >= size)
        return;
    int flag = 0;

    for (int i = 0; i < ndim; i++) {
        int index = ind / stride[i];
        if (index >= begin_pos[i] && index < end_pos[i])
            flag += 1;
        ind = ind % stride[i];
    }

    if (flag == ndim)
        output[tmp] = val;
    else
        output[tmp] = input[tmp];
}

int DLGpuSliceAssign(const DLArrayHandle input, DLArrayHandle output, float val,
                     int *begin_pos, int *end_pos,
                     DLStreamHandle stream_handle = NULL) {
    size_t ndim = input->ndim;
    size_t size = 1;
    for (int i = 0; i < ndim; ++i) {
        size *= input->shape[i];
    }
    const float *input_data = (const float *)input->data;
    float *output_data = (float *)output->data;

    int *begin = NULL;
    int *end = NULL;
    int64_t *stride = NULL;
    int dev_id = (input->ctx).device_id;
    if (is_chunk_init(dev_id) == false) {
        chunk_init(dev_id);
    }
    size_t pos_size = 3 * ndim * sizeof(int);
    size_t stride_size = 3 * ndim * sizeof(int64_t);
    begin = (int *)find_chunk(pos_size, dev_id);
    end = (int *)find_chunk(pos_size, dev_id);
    stride = (int64_t *)find_chunk(stride_size, dev_id);

    dim3 blocks;
    dim3 threads;
    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }
    cudaStream_t cu_stream = *(cudaStream_t *)(stream_handle->handle);
    if (cu_stream) {
        CUDA_CALL(cudaMemcpyAsync(begin, (void *)begin_pos, pos_size,
                                  cudaMemcpyHostToDevice, cu_stream));
        CUDA_CALL(cudaMemcpyAsync(end, (void *)end_pos, pos_size,
                                  cudaMemcpyHostToDevice, cu_stream));
        CUDA_CALL(cudaMemcpyAsync(stride, (void *)(input->stride), stride_size,
                                  cudaMemcpyHostToDevice, cu_stream));

        slice_assign<<<blocks, threads, 0, cu_stream>>>(
            input_data, output_data, val, begin, end, stride, size, ndim);
    } else {
        CUDA_CALL(cudaMemcpyAsync(begin, (void *)begin_pos, pos_size,
                                  cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpyAsync(end, (void *)end_pos, pos_size,
                                  cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpyAsync(stride, (void *)(input->stride), stride_size,
                                  cudaMemcpyHostToDevice));

        slice_assign<<<blocks, threads>>>(input_data, output_data, val, begin,
                                          end, stride, size, ndim);
    }
    return 0;
}

__global__ void slice_assign_matrix(const float *input_A, const float *input_B,
                                    float *output, int *begin_pos_A,
                                    int *end_pos_A, int *begin_pos_B,
                                    int64_t *stride_A, int64_t *stride_B,
                                    size_t size, size_t ndim) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    size_t tmp = ind;
    if (ind >= size)
        return;
    int flag = 0;
    int index_new = 0;
    for (int i = 0; i < ndim; i++) {
        int index = ind / stride_A[i];
        if (index >= begin_pos_A[i] && index < end_pos_A[i]) {
            flag += 1;
            index_new +=
                (index - begin_pos_A[i] + begin_pos_B[i]) * stride_B[i];
        }
        ind = ind % stride_A[i];
    }

    if (flag == ndim)
        output[tmp] = input_B[index_new];
    else
        output[tmp] = input_A[tmp];
}

int DLGpuSliceAssignMatrix(const DLArrayHandle input_A,
                           const DLArrayHandle input_B, DLArrayHandle output,
                           int *begin_pos_A, int *end_pos_A, int *begin_pos_B,
                           DLStreamHandle stream_handle = NULL) {
    size_t ndim = output->ndim;
    size_t size = 1;
    for (int i = 0; i < ndim; ++i) {
        size *= output->shape[i];
    }
    const float *input_A_data = (const float *)input_A->data;
    const float *input_B_data = (const float *)input_B->data;
    float *output_data = (float *)output->data;

    int *begin_A = NULL;
    int *begin_B = NULL;
    int *end_A = NULL;
    int64_t *stride_A = NULL;
    int64_t *stride_B = NULL;
    int dev_id = (input_A->ctx).device_id;
    if (is_chunk_init(dev_id) == false) {
        chunk_init(dev_id);
    }
    size_t pos_size = 3 * ndim * sizeof(int);
    size_t stride_size = 3 * ndim * sizeof(int64_t);
    begin_A = (int *)find_chunk(pos_size, dev_id);
    begin_B = (int *)find_chunk(pos_size, dev_id);
    end_A = (int *)find_chunk(pos_size, dev_id);
    stride_A = (int64_t *)find_chunk(stride_size, dev_id);
    stride_B = (int64_t *)find_chunk(stride_size, dev_id);

    dim3 blocks;
    dim3 threads;
    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }
    cudaStream_t cu_stream = *(cudaStream_t *)(stream_handle->handle);
    if (cu_stream) {
        CUDA_CALL(cudaMemcpyAsync(begin_A, (void *)begin_pos_A, pos_size,
                                  cudaMemcpyHostToDevice, cu_stream));
        CUDA_CALL(cudaMemcpyAsync(begin_B, (void *)begin_pos_B, pos_size,
                                  cudaMemcpyHostToDevice, cu_stream));
        CUDA_CALL(cudaMemcpyAsync(end_A, (void *)end_pos_A, pos_size,
                                  cudaMemcpyHostToDevice, cu_stream));
        CUDA_CALL(cudaMemcpyAsync(stride_A, (void *)(input_A->stride),
                                  stride_size, cudaMemcpyHostToDevice,
                                  cu_stream));
        CUDA_CALL(cudaMemcpyAsync(stride_B, (void *)(input_B->stride),
                                  stride_size, cudaMemcpyHostToDevice,
                                  cu_stream));

        slice_assign_matrix<<<blocks, threads, 0, cu_stream>>>(
            input_A_data, input_B_data, output_data, begin_A, end_A, begin_B,
            stride_A, stride_B, size, ndim);
    } else {
        CUDA_CALL(cudaMemcpyAsync(begin_A, (void *)begin_pos_A, pos_size,
                                  cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpyAsync(begin_B, (void *)begin_pos_B, pos_size,
                                  cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpyAsync(end_A, (void *)end_pos_A, pos_size,
                                  cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpyAsync(stride_A, (void *)(input_A->stride),
                                  stride_size, cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpyAsync(stride_B, (void *)(input_B->stride),
                                  stride_size, cudaMemcpyHostToDevice));

        slice_assign_matrix<<<blocks, threads>>>(
            input_A_data, input_B_data, output_data, begin_A, end_A, begin_B,
            stride_A, stride_B, size, ndim);
    }
    return 0;
}
