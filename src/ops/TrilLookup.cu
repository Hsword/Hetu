#include "gpu_runtime.h"

__global__ void tril_lookup_kernel(const float *input, float *output,
                                   int offset, size_t in_dim, size_t out_dim,
                                   size_t pre_size) {
    size_t ind_x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t ind_y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t ysize = in_dim * in_dim;
    if (ind_y >= pre_size || ind_x >= ysize)
        return;
    size_t cid = ind_x % in_dim;
    size_t rid = ind_x / in_dim;
    if (in_dim + rid + offset < in_dim + cid)
        return;
    size_t input_offset = ind_y * ysize + ind_x;
    size_t output_offset = ind_y * out_dim + cid;
    size_t start_ele = 1;
    size_t end_ele = rid;
    if (offset != 0) {
        if (offset < 0) {
            end_ele += offset;
        } else {
            start_ele += offset;
            end_ele += offset;
            if (end_ele > in_dim) {
                output_offset += (end_ele - in_dim) * in_dim;
                end_ele = in_dim;
            }
        }
    }
    output_offset += (start_ele + end_ele) * (end_ele - start_ele + 1) / 2;
    output[output_offset] = input[input_offset];
}

int DLGpuTrilLookup(const DLArrayHandle input, DLArrayHandle output, int offset,
                    DLStreamHandle stream_handle = NULL) {
    size_t pre_size = 1;
    size_t dim = input->ndim;
    assert(dim >= 2 && input->shape[dim - 2] == input->shape[dim - 1]);
    for (index_t i = 0; i < dim - 2; i++) {
        pre_size *= input->shape[i];
    }
    size_t mat_dim = input->shape[dim - 2];
    size_t mat_size = mat_dim * mat_dim;
    size_t out_dim = output->shape[output->ndim - 1];
    const float *input_data = (const float *)input->data;
    float *output_data = (float *)output->data;
    dim3 threads;
    dim3 blocks;
    ThreadBlock2D(threads, blocks, mat_size, pre_size);
    if (stream_handle)
        tril_lookup_kernel<<<blocks, threads, 0,
                             *(cudaStream_t *)stream_handle->handle>>>(
            input_data, output_data, offset, mat_dim, out_dim, pre_size);
    else
        tril_lookup_kernel<<<blocks, threads>>>(input_data, output_data, offset,
                                                mat_dim, out_dim, pre_size);
    return 0;
}

__global__ void tril_lookup_gradient_kernel(const float *input, float *output,
                                            int offset, size_t in_dim,
                                            size_t out_dim, size_t pre_size) {
    size_t ind_x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t ind_y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t ysize = out_dim * out_dim;
    if (ind_y >= pre_size || ind_x >= ysize)
        return;
    size_t cid = ind_x % out_dim;
    size_t rid = ind_x / out_dim;

    size_t output_offset = ind_y * ysize + ind_x;
    if (in_dim + rid + offset < in_dim + cid) {
        output[output_offset] = 0;
    } else {
        size_t input_offset = ind_y * in_dim + cid;
        size_t start_ele = 1;
        size_t end_ele = rid;
        if (offset != 0) {
            if (offset < 0) {
                end_ele += offset;
            } else {
                start_ele += offset;
                end_ele += offset;
                if (end_ele > out_dim) {
                    input_offset += (end_ele - out_dim) * out_dim;
                    end_ele = out_dim;
                }
            }
        }
        input_offset += (start_ele + end_ele) * (end_ele - start_ele + 1) / 2;
        output[output_offset] = input[input_offset];
    }
}

int DLGpuTrilLookupGradient(const DLArrayHandle input, DLArrayHandle output,
                            int offset, DLStreamHandle stream_handle = NULL) {
    size_t pre_size = 1;
    size_t dim = output->ndim;
    assert(dim >= 2 && output->shape[dim - 2] == output->shape[dim - 1]);
    for (index_t i = 0; i < dim - 2; i++) {
        pre_size *= output->shape[i];
    }
    size_t mat_dim = output->shape[dim - 2];
    size_t mat_size = mat_dim * mat_dim;
    size_t in_dim = input->shape[input->ndim - 1];
    const float *input_data = (const float *)input->data;
    float *output_data = (float *)output->data;
    dim3 threads;
    dim3 blocks;
    ThreadBlock2D(threads, blocks, mat_size, pre_size);
    if (stream_handle)
        tril_lookup_gradient_kernel<<<blocks, threads, 0,
                                      *(cudaStream_t *)stream_handle->handle>>>(
            input_data, output_data, offset, in_dim, mat_dim, pre_size);
    else
        tril_lookup_gradient_kernel<<<blocks, threads>>>(
            input_data, output_data, offset, in_dim, mat_dim, pre_size);
    return 0;
}
