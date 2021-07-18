#include "gpu_runtime.h"

__global__ void spmm_kernel(const int *indptr, const int *indices,
                            const float *data, const float *B, float *C, int k,
                            int n, int m, int start_pos, int end_pos) {
    // matC (n , m) matB (k , m) C = A * B
    // data & indices (nnz) ,indptr(n)
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    size_t ind = id / m;
    size_t offset = id - ind * m;
    if (ind >= n)
        return;
    C[m * ind + offset] = 0;
    int i_s = indptr[ind], i_e = indptr[ind + 1];
    if (start_pos == -1) {
        for (int i = i_s; i < i_e; i++) {
            int from = indices[i];
            float scale = data[i];
            C[m * ind + offset] += B[m * from + offset] * scale;
        }
    } else {
        for (int i = i_s; i < i_e; i++) {
            if (indices[i] >= start_pos && indices[i] < end_pos) {
                int from = indices[i] - start_pos;
                float scale = data[i];
                C[m * ind + offset] += B[m * from + offset] * scale;
            }
        }
    }
    return;
}

__global__ void spmm_set_zero_kernel(float *output, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    output[ind] = 0;
}

__global__ void spmm_T_kernel(const int *indptr, const int *indices,
                              const float *data, const float *B, float *C,
                              int k, int n, int m) {
    // matC (n , m) matB (k , m) C = A^T *B
    // data & indices (nnz) ,indptr(k)
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    size_t ind = id / m;
    size_t offset = id - ind * m;
    if (ind >= k)
        return;
    // C[m * ind + offset] = 0;
    int i_s = indptr[ind], i_e = indptr[ind + 1];
    float val = B[m * ind + offset];
    for (int i = i_s; i < i_e; i++) {
        int to = indices[i];
        float addend = data[i] * val;
        atomicAdd(&C[m * to + offset], addend);
    }
    return;
}

int CuSparse_DLGpuCsrmm0(const DLArrayHandle data_handle,
                         const DLArrayHandle row_handle,
                         const DLArrayHandle col_handle, int nrow, int ncol,
                         const DLArrayHandle matB, DLArrayHandle matC,
                         int start_pos, int end_pos,
                         DLStreamHandle stream_handle = NULL) {
    int n = matC->shape[0];
    int m = matC->shape[1];
    int k = matB->shape[0];
    dim3 blocks;
    dim3 threads;
    if (n * m <= 1024) {
        threads.x = n * m;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (n * m + 1023) / 1024;
    }

    if (stream_handle) {
        spmm_kernel<<<blocks, threads, 0,
                      *(cudaStream_t *)stream_handle->handle>>>(
            (const int *)row_handle->data, (const int *)col_handle->data,
            (const float *)data_handle->data, (const float *)matB->data,
            (float *)matC->data, k, n, m, start_pos, end_pos);
    } else {
        spmm_kernel<<<blocks, threads>>>(
            (const int *)row_handle->data, (const int *)col_handle->data,
            (const float *)data_handle->data, (const float *)matB->data,
            (float *)matC->data, k, n, m, start_pos, end_pos);
    }

    return 0;
}

int CuSparse_DLGpuCsrmm1(const DLArrayHandle data_handle,
                         const DLArrayHandle row_handle,
                         const DLArrayHandle col_handle, int nrow, int ncol,
                         const DLArrayHandle matB, DLArrayHandle matC,
                         DLStreamHandle stream_handle = NULL) {
    int n = matC->shape[0];
    int m = matC->shape[1];
    int k = matB->shape[0];
    dim3 blocks;
    dim3 threads;
    if (k * m <= 1024) {
        threads.x = k * m;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (k * m + 1023) / 1024;
    }

    if (stream_handle) {
        spmm_set_zero_kernel<<<n, m, 0,
                               *(cudaStream_t *)stream_handle->handle>>>(
            (float *)matC->data, n * m);
        spmm_T_kernel<<<blocks, threads, 0,
                        *(cudaStream_t *)stream_handle->handle>>>(
            (const int *)row_handle->data, (const int *)col_handle->data,
            (const float *)data_handle->data, (const float *)matB->data,
            (float *)matC->data, k, n, m);
    } else {
        spmm_set_zero_kernel<<<n, m>>>((float *)matC->data, n * m);
        spmm_T_kernel<<<blocks, threads>>>(
            (const int *)row_handle->data, (const int *)col_handle->data,
            (const float *)data_handle->data, (const float *)matB->data,
            (float *)matC->data, k, n, m);
    }

    return 0;
}

int CuSparse_DLGpuCsrmm(const DLArrayHandle data_handle,
                        const DLArrayHandle row_handle,
                        const DLArrayHandle col_handle, int nrow, int ncol,
                        bool transposeA, const DLArrayHandle matB,
                        bool transposeB, DLArrayHandle matC, int start_pos = -1,
                        int end_pos = -1, DLStreamHandle stream_handle = NULL) {
    assert(!transposeB);
    assert(data_handle->ndim == 1);
    assert(row_handle->ndim == 1);
    assert(col_handle->ndim == 1);
    assert(matB->ndim == 2);
    assert(matC->ndim == 2);
    if (!transposeA) {
        return CuSparse_DLGpuCsrmm0(data_handle, row_handle, col_handle, nrow,
                                    ncol, matB, matC, start_pos, end_pos,
                                    stream_handle);
    } else {
        return CuSparse_DLGpuCsrmm1(data_handle, row_handle, col_handle, nrow,
                                    ncol, matB, matC, stream_handle);
    }
}