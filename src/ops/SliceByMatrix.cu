#include "gpu_runtime.h"

__global__ void slice_by_matrix_kernel(const float *input, const float *index1,
                                       const float *index2, float *output,
                                       int nb, int nc, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;

    int nrow = ind / nc;
    int ncol = ind % nc;
    int index = int(index1[nrow]) * nb * nc + int(index2[nrow]) * nc + ncol;
    output[ind] = input[index];
}

int DLGpuSliceByMatrix(const DLArrayHandle in_arr, const DLArrayHandle index1,
                       const DLArrayHandle index2, DLArrayHandle out_arr,
                       DLStreamHandle stream_handle = NULL) {
    assert(in_arr->ndim == 3);
    size_t ndim = out_arr->ndim;
    size_t size = 1;
    for (int i = 0; i < ndim; ++i) {
        size *= out_arr->shape[i];
    }
    int nb = in_arr->shape[1];
    int nc = in_arr->shape[2];

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

    const float *in_arr_data = (const float *)(in_arr->data);
    const float *index1_data = (const float *)(index1->data);
    const float *index2_data = (const float *)(index2->data);
    float *out_arr_data = (float *)(out_arr->data);

    if (cu_stream) {
        slice_by_matrix_kernel<<<blocks, threads, 0, cu_stream>>>(
            in_arr_data, index1_data, index2_data, out_arr_data, nb, nc, size);

    } else {
        slice_by_matrix_kernel<<<blocks, threads>>>(
            in_arr_data, index1_data, index2_data, out_arr_data, nb, nc, size);
    }
    return 0;
}

__global__ void slice_by_matrix_gradient_kernel(const float *input,
                                                const float *index1,
                                                const float *index2,
                                                float *output, int nb, int nc,
                                                size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    float val = input[ind];
    int nrow = ind / nc;
    int ncol = ind % nc;
    int index = int(index1[nrow]) * nb * nc + int(index2[nrow]) * nc + ncol;
    atomicAdd(&output[index], val);
}

int DLGpuSliceByMatrixGradient(const DLArrayHandle in_arr,
                               const DLArrayHandle index1,
                               const DLArrayHandle index2,
                               DLArrayHandle out_arr,
                               DLStreamHandle stream_handle = NULL) {
    assert(out_arr->ndim == 3);
    size_t ndim = in_arr->ndim;
    size_t size = 1;
    for (int i = 0; i < ndim; ++i) {
        size *= in_arr->shape[i];
    }
    int nb = out_arr->shape[1];
    int nc = out_arr->shape[2];

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

    const float *in_arr_data = (const float *)(in_arr->data);
    const float *index1_data = (const float *)(index1->data);
    const float *index2_data = (const float *)(index2->data);
    float *out_arr_data = (float *)(out_arr->data);

    if (cu_stream) {
        slice_by_matrix_gradient_kernel<<<blocks, threads, 0, cu_stream>>>(
            in_arr_data, index1_data, index2_data, out_arr_data, nb, nc, size);

    } else {
        slice_by_matrix_gradient_kernel<<<blocks, threads>>>(
            in_arr_data, index1_data, index2_data, out_arr_data, nb, nc, size);
    }
    return 0;
}