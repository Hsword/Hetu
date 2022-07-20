#include "gpu_runtime.h"

__global__ void broadcast_shape_kernel(const float *input_data,
                                       float *output_data, uint *out_strides,
                                       uint *in_dims, size_t ndims,
                                       size_t output_size) {
    size_t o_ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (o_ind >= output_size)
        return;
    size_t i_ind = 0;
    uint temp = o_ind;
    for (int i = 0; i < ndims; ++i) {
        i_ind *= in_dims[i];
        i_ind += (in_dims[i] > 1) * temp / out_strides[i];
        temp %= out_strides[i];
    }
    output_data[o_ind] = input_data[i_ind];
}

int DLGpuBroadcastShape(const DLArrayHandle in_arr, DLArrayHandle out_arr,
                        int *add_axes, DLStreamHandle stream_handle = NULL) {
    size_t allocated = out_arr->ndim * sizeof(uint);

    uint *out_strides = (uint *)malloc(allocated);
    uint *in_dims = (uint *)malloc(allocated);

    size_t output_size = 1;
    size_t diff = out_arr->ndim - in_arr->ndim;

    if (add_axes == NULL) {
        for (int i = out_arr->ndim - 1; i >= 0; --i) {
            out_strides[i] = output_size;
            output_size *= out_arr->shape[i];
            if (i < diff) {
                in_dims[i] = 1;
            } else {
                in_dims[i] = in_arr->shape[i - diff];
            }
        }
    } else {
        for (int i = out_arr->ndim - 1; i >= 0; --i) {
            out_strides[i] = output_size;
            output_size *= out_arr->shape[i];
            in_dims[i] = 0;
        }
        for (int i = 0; i < diff; ++i) {
            in_dims[add_axes[i]] = 1;
        }
        int o_ind = 0;
        for (int i = 0; i < in_arr->ndim; ++i) {
            while (in_dims[o_ind++] == 1)
                ;
            in_dims[o_ind - 1] = in_arr->shape[i];
        }
    }

    int dev_id = (in_arr->ctx).device_id;
    if (is_chunk_init(dev_id) == false) {
        chunk_init(dev_id);
    }
    uint *gpu_strides = (uint *)find_chunk(allocated, dev_id);
    uint *gpu_dims = (uint *)find_chunk(allocated, dev_id);

    dim3 blocks;
    dim3 threads;
    if (output_size <= 1024) {
        threads.x = output_size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (output_size + 1023) / 1024;
    }

    cudaStream_t cu_stream = static_cast<cudaStream_t>(
        stream_handle ? *(cudaStream_t *)(stream_handle->handle) : NULL);

    if (cu_stream != NULL) {
        CUDA_CALL(cudaMemcpyAsync(gpu_strides, out_strides, allocated,
                                  cudaMemcpyHostToDevice, cu_stream));
        CUDA_CALL(cudaMemcpyAsync(gpu_dims, in_dims, allocated,
                                  cudaMemcpyHostToDevice, cu_stream));
        broadcast_shape_kernel<<<blocks, threads, 0, cu_stream>>>(
            (const float *)(in_arr->data), (float *)(out_arr->data),
            gpu_strides, gpu_dims, (size_t)out_arr->ndim, output_size);
    } else {
        CUDA_CALL(cudaMemcpy(gpu_strides, out_strides, allocated,
                             cudaMemcpyHostToDevice));
        CUDA_CALL(
            cudaMemcpy(gpu_dims, in_dims, allocated, cudaMemcpyHostToDevice));
        broadcast_shape_kernel<<<blocks, threads>>>(
            (const float *)(in_arr->data), (float *)(out_arr->data),
            gpu_strides, gpu_dims, (size_t)out_arr->ndim, output_size);
    }

    del_chunk(gpu_strides, dev_id);
    del_chunk(gpu_dims, dev_id);
    free(out_strides);
    free(in_dims);
    return 0;
}

int DLGpuBroadcastShapeSimple(const DLArrayHandle in_arr, DLArrayHandle out_arr,
                              const DLArrayHandle out_strides,
                              const DLArrayHandle in_dims,
                              DLStreamHandle stream_handle) {
    const float *in_data = (const float *)(in_arr->data);
    float *out_data = (float *)(out_arr->data);
    uint *os_data = (uint *)(out_strides->data);
    uint *id_data = (uint *)(in_dims->data);
    size_t output_size = 1;
    for (size_t i = 0; i < out_arr->ndim; ++i) {
        output_size *= out_arr->shape[i];
    }

    dim3 blocks;
    dim3 threads;
    if (output_size <= 1024) {
        threads.x = output_size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (output_size + 1023) / 1024;
    }
    size_t ndim = (size_t)in_dims->shape[0];

    cudaStream_t cu_stream = static_cast<cudaStream_t>(
        stream_handle ? *(cudaStream_t *)(stream_handle->handle) : NULL);

    if (cu_stream != NULL) {
        broadcast_shape_kernel<<<blocks, threads, 0, cu_stream>>>(
            in_data, out_data, os_data, id_data, ndim, output_size);
    } else {
        broadcast_shape_kernel<<<blocks, threads>>>(in_data, out_data, os_data,
                                                    id_data, ndim, output_size);
    }

    return 0;
}
