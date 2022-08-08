#include "gpu_runtime.h"

__global__ void sparse_embedding_lookup_kernel(const float *value,
                                               const int *rows, const int *cols,
                                               const int *ids, float *output,
                                               size_t size, size_t length) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size)
        return;
    int rowid = ids[index];
    int left = rows[rowid];
    int right = rows[rowid + 1];
    float *pout = output + index * length;
    for (int i = 0; i < length; ++i) {
        pout[i] = 0;
    }
    for (int i = left; i < right; ++i) {
        pout[cols[i]] = value[i];
    }
}

int DLGpuSparseEmbeddingLookUp(const DLArrayHandle value,
                               const DLArrayHandle row, const DLArrayHandle col,
                               const DLArrayHandle ids, DLArrayHandle output,
                               DLStreamHandle stream_handle = NULL) {
    size_t size = ArrSize(ids);
    size_t length = output->shape[output->ndim - 1];
    const float *value_data = (const float *)value->data;
    const int *row_data = (const int *)row->data;
    const int *col_data = (const int *)col->data;
    const int *ids_data = (const int *)ids->data;
    float *output_data = (float *)output->data;
    dim3 blocks;
    dim3 threads;
    ThreadBlock1D(threads, blocks, size);
    if (stream_handle)
        sparse_embedding_lookup_kernel<<<
            blocks, threads, 0, *(cudaStream_t *)stream_handle->handle>>>(
            value_data, row_data, col_data, ids_data, output_data, size,
            length);
    else
        sparse_embedding_lookup_kernel<<<blocks, threads>>>(
            value_data, row_data, col_data, ids_data, output_data, size,
            length);
    return 0;
}
