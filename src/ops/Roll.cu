#include "gpu_runtime.h"

__global__ void roll_kernel(const float *input, float *output, int N, int rank,
                            uint *shifts, uint *strides, uint *sizes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) {
        return;
    }

    int output_idx = idx;
    int new_dim_idx = 0;

#pragma unroll
    for (int i = 0; i < rank; i++) {
        new_dim_idx = (idx / strides[i]) % sizes[i] + shifts[i];
        if (new_dim_idx >= sizes[i])
            output_idx += (shifts[i] - sizes[i]) * strides[i];
        else
            output_idx += shifts[i] * strides[i];
    }
    output[output_idx] = input[idx];
}

int DLGpuRoll(const DLArrayHandle input, int *shift, int *axis, int nums,
              DLArrayHandle output, DLStreamHandle stream_handle = NULL) {
    const float *input_data = (const float *)input->data;
    float *output_data = (float *)output->data;

    int len = 1;
    int n_dims = input->ndim;

    int *stride_dim = new int[n_dims];
    stride_dim[n_dims - 1] = 1;
    for (int i = 0; i < n_dims; i++) {
        len *= input->shape[i];
        if (i > 0)
            stride_dim[n_dims - i - 1] =
                input->shape[n_dims - i] * stride_dim[n_dims - i];
    }

    int *strides = new int[nums];
    int *sizes = new int[nums];
    int *shifts = new int[nums];

    if (axis == NULL) {
        strides[0] = 1;
        sizes[0] = len;
        shift[0] = (shift[0] % len + len) % len;
    } else {
        for (int i = 0; i < nums; i++) {
            int dim = axis[i] >= 0 ? axis[0] : axis[i] + n_dims;
            int size = input->shape[dim];
            if (size != 0) {
                strides[i] = stride_dim[i];
                sizes[i] = size;
                shift[i] = (shift[i] % size + size) % size;
            }
        }
    }

    uint *shifts_buf = NULL;
    uint *strides_buf = NULL;
    uint *sizes_buf = NULL;
    cudaStream_t cu_stream = static_cast<cudaStream_t>(
        stream_handle ? *(cudaStream_t *)(stream_handle->handle) : NULL);

    int dev_id = (input->ctx).device_id;
    if (is_chunk_init(dev_id) == false) {
        chunk_init(dev_id);
    }
    size_t buf_size = nums * sizeof(uint);
    shifts_buf = (uint *)find_chunk(buf_size, dev_id);
    strides_buf = (uint *)find_chunk(buf_size, dev_id);
    sizes_buf = (uint *)find_chunk(buf_size, dev_id);
    CUDA_CALL(cudaMemcpyAsync(shifts_buf, (void *)shift, buf_size,
                              cudaMemcpyHostToDevice, cu_stream));
    CUDA_CALL(cudaMemcpyAsync(strides_buf, (void *)strides, buf_size,
                              cudaMemcpyHostToDevice, cu_stream));
    CUDA_CALL(cudaMemcpyAsync(sizes_buf, (void *)sizes, buf_size,
                              cudaMemcpyHostToDevice, cu_stream));

    dim3 blocks;
    dim3 threads;

    if (len <= 1024) {
        threads.x = len;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (len + 1023) / 1024;
    }
    if (stream_handle)
        roll_kernel<<<blocks, threads, 0, cu_stream>>>(input_data, output_data,
                                                       len, nums, shifts_buf,
                                                       strides_buf, sizes_buf);
    else
        roll_kernel<<<blocks, threads>>>(input_data, output_data, len, nums,
                                         shifts_buf, strides_buf, sizes_buf);
    return 0;
}
