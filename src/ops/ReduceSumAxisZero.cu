#include "gpu_runtime.h"

__global__ void float_memory_copy(float *A, const float *B, size_t len) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= len)
        return;
    A[id] = B[id];
}
__global__ void float_add_kernel(float *A, const float *B, size_t len) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= len)
        return;
    A[id] += B[id];
}

int Float_Add(float *A, const float *B, int len, DLStreamHandle stream_handle) {
    size_t BLOCKS = (len + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (stream_handle)
        float_add_kernel<<<BLOCKS, THREADS_PER_BLOCK, 0,
                           *(cudaStream_t *)stream_handle->handle>>>(A, B, len);
    else
        float_add_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(A, B, len);
    return 0;
}

int DLGpuReduceSumAxisZero(const DLArrayHandle input, DLArrayHandle output,
                           DLStreamHandle stream_handle = NULL) {
    for (int i = 1; i < (input->ndim); i++) {
        assert((input->shape[i]) == (output->shape[i - 1]));
    }
    const float *input_data = (const float *)input->data;

    int dev_id = (input->ctx).device_id;
    size_t workspace_size = (input->shape[0] + 1) / 2;
    for (int i = 1; i < (input->ndim); i++) {
        workspace_size *= input->shape[i];
    }
    if (is_chunk_init(dev_id) == false) {
        chunk_init(dev_id);
    }

    void *work_data = find_chunk(workspace_size * sizeof(float), dev_id);
    float *workspace_data = (float *)work_data;

    float *output_data = (float *)output->data;
    size_t output_size = 1;
    for (index_t i = 0; i < (output->ndim); i++) {
        output_size *= (output->shape[i]);
    }
    size_t batch = input->shape[0];
    size_t BLOCKS = ((batch + 1) / 2 * output_size + THREADS_PER_BLOCK - 1)
                    / THREADS_PER_BLOCK;
    if (stream_handle)
        float_memory_copy<<<BLOCKS, THREADS_PER_BLOCK, 0,
                            *(cudaStream_t *)stream_handle->handle>>>(
            workspace_data, input_data, (batch + 1) / 2 * output_size);
    else
        float_memory_copy<<<BLOCKS, THREADS_PER_BLOCK>>>(
            workspace_data, input_data, (batch + 1) / 2 * output_size);
    Float_Add(workspace_data, input_data + (batch + 1) / 2 * output_size,
              batch / 2 * output_size, stream_handle);
    batch = (batch + 1) / 2;
    while (batch != 1) {
        Float_Add(workspace_data,
                  workspace_data + (batch + 1) / 2 * output_size,
                  batch / 2 * output_size, stream_handle);
        batch = (batch + 1) / 2;
    }
    BLOCKS = (output_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (stream_handle)
        float_memory_copy<<<BLOCKS, THREADS_PER_BLOCK, 0,
                            *(cudaStream_t *)stream_handle->handle>>>(
            output_data, workspace_data, output_size);
    else
        float_memory_copy<<<BLOCKS, THREADS_PER_BLOCK>>>(
            output_data, workspace_data, output_size);
    del_chunk(work_data, dev_id);
    return 0;
}

__global__ void tree_reduce_kernel(const size_t nblocks, const float *input,
                                   float *output, size_t now_batch,
                                   size_t output_size, size_t total_len) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nblocks)
        return;
    size_t row = idx / output_size;
    size_t col = idx % output_size;
    row = row * 2 * now_batch * output_size;
    if (row + col < total_len
        && row + col + now_batch * output_size < total_len) {
        output[row + col] =
            input[row + col] + input[row + col + now_batch * output_size];
        output[row + col + now_batch * output_size] =
            input[row + col + now_batch * output_size];
    }
    if (row + col < total_len
        && row + col + now_batch * output_size >= total_len) {
        output[row + col] = input[row + col];
    }
}
int _DLGpuReduceSumAxisZero(const DLArrayHandle input, DLArrayHandle output,
                            DLArrayHandle arr_workspace,
                            DLStreamHandle stream_handle = NULL) {
    const float *input_data = (const float *)(input->data);
    float *workspace_data = (float *)(arr_workspace->data);
    float *output_data = (float *)(output->data);
    size_t output_size = 1;
    size_t input_size = 1;
    for (index_t i = 0; i < (output->ndim); i++) {
        output_size *= (output->shape[i]);
    }
    input_size = output_size * input->shape[0];
    size_t batch = input->shape[0];
    size_t BLOCKS = ((batch + 1) / 2 * output_size + THREADS_PER_BLOCK - 1)
                    / THREADS_PER_BLOCK;
    size_t now = 1;
    if (stream_handle)
        tree_reduce_kernel<<<BLOCKS, THREADS_PER_BLOCK, 0,
                             *(cudaStream_t *)stream_handle->handle>>>(
            (batch + 1) / 2 * output_size, input_data, workspace_data, now,
            output_size, input_size);
    else
        tree_reduce_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(
            (batch + 1) / 2 * output_size, input_data, workspace_data, now,
            output_size, input_size);
    now *= 2;
    batch = (batch + 1) / 2;
    while (now < (input->shape[0])) {
        BLOCKS = ((batch + 1) / 2 * output_size + THREADS_PER_BLOCK - 1)
                 / THREADS_PER_BLOCK;
        if (stream_handle)
            tree_reduce_kernel<<<BLOCKS, THREADS_PER_BLOCK, 0,
                                 *(cudaStream_t *)stream_handle->handle>>>(
                (batch + 1) / 2 * output_size, workspace_data, workspace_data,
                now, output_size, input_size);
        else
            tree_reduce_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(
                (batch + 1) / 2 * output_size, workspace_data, workspace_data,
                now, output_size, input_size);
        now = now * 2;
        batch = (batch + 1) / 2;
    }
    BLOCKS = (output_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (stream_handle)
        float_memory_copy<<<BLOCKS, THREADS_PER_BLOCK, 0,
                            *(cudaStream_t *)stream_handle->handle>>>(
            output_data, workspace_data, output_size);
    else
        float_memory_copy<<<BLOCKS, THREADS_PER_BLOCK>>>(
            output_data, workspace_data, output_size);
    return 0;
}
