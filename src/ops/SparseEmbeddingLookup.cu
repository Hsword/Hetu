#include "gpu_runtime.h"

__global__ void csr_embedding_lookup_kernel(const float *value, const int *rows,
                                            const int *cols, const int *ids,
                                            float *output, size_t size,
                                            size_t dim) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size)
        return;
    int rowid = ids[index];
    int left = rows[rowid];
    int right = rows[rowid + 1];
    float *pout = output + index * dim;
    for (int i = 0; i < dim; ++i) {
        pout[i] = 0;
    }
    for (int i = left; i < right; ++i) {
        pout[cols[i]] = value[i];
    }
}

int DLGpuCSREmbeddingLookUp(const DLArrayHandle value, const DLArrayHandle row,
                            const DLArrayHandle col, const DLArrayHandle ids,
                            DLArrayHandle output,
                            DLStreamHandle stream_handle = NULL) {
    size_t size = ArrSize(ids);
    size_t dim = output->shape[output->ndim - 1];
    const float *value_data = (const float *)value->data;
    const int *row_data = (const int *)row->data;
    const int *col_data = (const int *)col->data;
    const int *ids_data = (const int *)ids->data;
    float *output_data = (float *)output->data;
    dim3 blocks;
    dim3 threads;
    ThreadBlock1D(threads, blocks, size);
    if (stream_handle)
        csr_embedding_lookup_kernel<<<blocks, threads, 0,
                                      *(cudaStream_t *)stream_handle->handle>>>(
            value_data, row_data, col_data, ids_data, output_data, size, dim);
    else
        csr_embedding_lookup_kernel<<<blocks, threads>>>(
            value_data, row_data, col_data, ids_data, output_data, size, dim);
    return 0;
}

__global__ void coo_embedding_lookup_kernel(const float *value, const int *rows,
                                            const int *cols, const int *ids,
                                            float *output, size_t size,
                                            size_t length, size_t dim) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size)
        return;
    int rowid = ids[index];

    float *pout = output + index * dim;
    for (int i = 0; i < dim; ++i) {
        pout[i] = 0;
    }
    // binary search
    if (rows[0] <= rowid && rowid <= rows[length - 1]) {
        size_t left = 0, right = length, middle = 0;
        if (rows[0] < rowid) {
            while (!(rows[left] < rowid && rows[left + 1] >= rowid)) {
                middle = (left + right) / 2;
                if (rows[middle] >= rowid)
                    right = middle;
                else
                    left = middle;
            }
            ++left;
        }
        while (left < length && rows[left] == rowid) {
            pout[cols[left]] = value[left];
            ++left;
        }
    }
}

int DLGpuCOOEmbeddingLookUp(const DLArrayHandle value, const DLArrayHandle row,
                            const DLArrayHandle col, const DLArrayHandle ids,
                            DLArrayHandle output,
                            DLStreamHandle stream_handle = NULL) {
    size_t size = ArrSize(ids);
    size_t dim = output->shape[output->ndim - 1];
    size_t length = row->shape[0];
    const float *value_data = (const float *)value->data;
    const int *row_data = (const int *)row->data;
    const int *col_data = (const int *)col->data;
    const int *ids_data = (const int *)ids->data;
    float *output_data = (float *)output->data;
    dim3 blocks;
    dim3 threads;
    ThreadBlock1D(threads, blocks, size);
    if (stream_handle)
        coo_embedding_lookup_kernel<<<blocks, threads, 0,
                                      *(cudaStream_t *)stream_handle->handle>>>(
            value_data, row_data, col_data, ids_data, output_data, size, length,
            dim);
    else
        coo_embedding_lookup_kernel<<<blocks, threads>>>(
            value_data, row_data, col_data, ids_data, output_data, size, length,
            dim);
    return 0;
}
