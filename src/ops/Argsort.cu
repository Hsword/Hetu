#include "gpu_runtime.h"
#include <cub/cub.cuh>

struct OffsetIter {
    OffsetIter(int ncol_) : ncol(ncol_) {
    }
    __host__ __device__ __forceinline__ int operator()(const int &idx) const {
        return idx * ncol;
    }
    int ncol;
};

__global__ void set_index(float *indices, int num_rows, int num_cols) {
    int col_id = threadIdx.x;
    int row_id = blockIdx.x;

    for (int j = row_id; j < num_rows; j += gridDim.x) {
        for (int i = col_id; i < num_cols; i += blockDim.x) {
            indices[j * num_cols + i] = i;
        }
    }
}

int DLGpuArgsort(const DLArrayHandle input, DLArrayHandle output,
                 DLArrayHandle index, DLArrayHandle output_index, int dim,
                 bool descending, DLStreamHandle stream_handle = NULL) {
    const int kThreadsPerBlock = 1024;
    int n_dims = input->ndim;
    int axis = (dim < 0) ? (n_dims + dim) : dim;
    assert(axis == 1);
    assert(n_dims == 2);

    int nrow = input->shape[0];
    int ncol = input->shape[1];
    int blocks = nrow;
    int threads = ceil(ncol / 64.0) * 64;
    threads = threads > kThreadsPerBlock ? kThreadsPerBlock : threads;

    const float *input_data = (const float *)input->data;
    float *output_data = (float *)output->data;
    float *index_data = (float *)index->data;
    float *output_index_data = (float *)output_index->data;

    cudaStream_t cu_stream = NULL;

    if (stream_handle) {
        cu_stream = (*(cudaStream_t *)(stream_handle->handle));
        set_index<<<blocks, threads, 0, cu_stream>>>(index_data, nrow, ncol);
    } else
        set_index<<<blocks, threads>>>(index_data, nrow, ncol);

    cub::CountingInputIterator<int> counting_iter(0);
    cub::TransformInputIterator<int, OffsetIter,
                                cub::CountingInputIterator<int>>
        segment_offsets_t(counting_iter, OffsetIter(ncol));

    size_t temp_storage_bytes = 0;
    void *d_temp_storage = NULL;

    if (descending) {
        CUDA_CALL(cub::DeviceSegmentedRadixSort::SortPairsDescending(
            d_temp_storage, temp_storage_bytes, input_data, output_data,
            index_data, output_index_data, nrow * ncol, nrow, segment_offsets_t,
            segment_offsets_t + 1, 0, sizeof(float) * 8, cu_stream));
    } else {
        CUDA_CALL(cub::DeviceSegmentedRadixSort::SortPairs(
            d_temp_storage, temp_storage_bytes, input_data, output_data,
            index_data, output_index_data, nrow * ncol, nrow, segment_offsets_t,
            segment_offsets_t + 1, 0, sizeof(float) * 8, cu_stream));
    }
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    if (descending) {
        CUDA_CALL(cub::DeviceSegmentedRadixSort::SortPairsDescending(
            d_temp_storage, temp_storage_bytes, input_data, output_data,
            index_data, output_index_data, nrow * ncol, nrow, segment_offsets_t,
            segment_offsets_t + 1, 0, sizeof(float) * 8, cu_stream));
    } else {
        CUDA_CALL(cub::DeviceSegmentedRadixSort::SortPairs(
            d_temp_storage, temp_storage_bytes, input_data, output_data,
            index_data, output_index_data, nrow * ncol, nrow, segment_offsets_t,
            segment_offsets_t + 1, 0, sizeof(float) * 8, cu_stream));
    }

    return 0;
}
