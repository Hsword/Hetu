#include <nppdefs.h>
#include "gpu_runtime.h"
#include "gpu_functions.cuh"
#include "random.h"

__global__ void assign_with_indexedslices_kernel(float *embedding,
                                                 const int *indices,
                                                 const float *values,
                                                 size_t nrow, size_t dim,
                                                 size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    int index = indices[ind];
    if (index >= 0 && index < nrow) {
        float *emb_ptr = embedding + index * dim;
        const float *val_ptr = values + ind * dim;
        for (int i = 0; i < dim; ++i)
            emb_ptr[i] = val_ptr[i];
    }
}

int DLGpuAssignWithIndexedSlices(DLArrayHandle embedding,
                                 const DLArrayHandle indices,
                                 const DLArrayHandle values,
                                 DLStreamHandle stream_handle = NULL) {
    size_t size = ArrSize(indices);
    dim3 blocks;
    dim3 threads;
    ThreadBlock1D(threads, blocks, size);
    float *embed_data = (float *)embedding->data;
    const int *indices_data = (const int *)indices->data;
    const float *values_data = (const float *)values->data;
    assert(embedding->ndim == 2);
    size_t nrow = embedding->shape[0];
    size_t dim = embedding->shape[1];
    if (stream_handle) {
        assign_with_indexedslices_kernel<<<
            blocks, threads, 0, *(cudaStream_t *)stream_handle->handle>>>(
            embed_data, indices_data, values_data, nrow, dim, size);
    } else {
        assign_with_indexedslices_kernel<<<blocks, threads>>>(
            embed_data, indices_data, values_data, nrow, dim, size);
    }
    return 0;
}

template <class T>
__global__ void assign_quantized_embedding_unified_kernel(
    T *embedding, const int *indices, const float *values, float scale,
    float minele, HetuRandomState cudars, bool stochastic, size_t nrow,
    size_t dim, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    int index = indices[ind];
    if (index >= 0 && index < nrow) {
        T *tar_ptr = embedding + index * dim;
        const float *src_ptr = values + ind * dim;
        for (size_t i = 0; i < dim; ++i) {
            float cur_value = src_ptr[i];
            T out;
            if (stochastic) {
                out = stochastic_rounding<T>(cur_value, scale, minele, cudars,
                                             ind);
            } else {
                out = fixed_rounding<T>(cur_value, scale, minele);
            }
            tar_ptr[i] = out;
        }
    }
}

template <class T>
__global__ void
assign_quantized_embedding_kernel(T *embedding, const int *indices,
                                  const float *values, const float *qparam,
                                  HetuRandomState cudars, bool stochastic,
                                  size_t nrow, size_t dim, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    int index = indices[ind];
    if (index >= 0 && index < nrow) {
        T *tar_ptr = embedding + index * dim;
        const float *src_ptr = values + ind * dim;
        const float *cur_qp = qparam + index * 2;
        float scale = cur_qp[0];
        float minele = cur_qp[1];
        for (size_t i = 0; i < dim; ++i) {
            float cur_value = src_ptr[i];
            T out;
            if (stochastic) {
                out = stochastic_rounding<T>(cur_value, scale, minele, cudars,
                                             ind);
            } else {
                out = fixed_rounding<T>(cur_value, scale, minele);
            }
            tar_ptr[i] = out;
        }
    }
}

int DLGpuAssignQuantizedEmbeddingUnified(DLArrayHandle embedding,
                                         const DLArrayHandle indices,
                                         const DLArrayHandle values,
                                         float scale, float minele, int digit,
                                         bool stochastic,
                                         DLStreamHandle stream_handle = NULL) {
    size_t size = ArrSize(indices);
    dim3 blocks;
    dim3 threads;
    ThreadBlock1D(threads, blocks, size);
    const int *indices_data = (const int *)indices->data;
    const float *values_data = (const float *)values->data;
    assert(embedding->ndim == 2);
    size_t nrow = embedding->shape[0];
    size_t dim = embedding->shape[1];
    HetuRandomState &cudars = GetRandomState(dim);
    if (digit == 16) {
        uint16_t *embed_data = (uint16_t *)embedding->data;

        if (stream_handle) {
            assign_quantized_embedding_unified_kernel<uint16_t>
                <<<blocks, threads, 0,
                   *(cudaStream_t *)stream_handle->handle>>>(
                    embed_data, indices_data, values_data, scale, minele,
                    cudars, stochastic, nrow, dim, size);
        } else {
            assign_quantized_embedding_unified_kernel<uint16_t>
                <<<blocks, threads>>>(embed_data, indices_data, values_data,
                                      scale, minele, cudars, stochastic, nrow,
                                      dim, size);
        }
    } else {
        uint8_t *embed_data = (uint8_t *)embedding->data;

        if (stream_handle) {
            assign_quantized_embedding_unified_kernel<uint8_t>
                <<<blocks, threads, 0,
                   *(cudaStream_t *)stream_handle->handle>>>(
                    embed_data, indices_data, values_data, scale, minele,
                    cudars, stochastic, nrow, dim, size);
        } else {
            assign_quantized_embedding_unified_kernel<uint8_t>
                <<<blocks, threads>>>(embed_data, indices_data, values_data,
                                      scale, minele, cudars, stochastic, nrow,
                                      dim, size);
        }
    }
    return 0;
}

int DLGpuAssignQuantizedEmbedding(DLArrayHandle embedding,
                                  const DLArrayHandle indices,
                                  const DLArrayHandle values,
                                  const DLArrayHandle qparam, int digit,
                                  bool stochastic,
                                  DLStreamHandle stream_handle = NULL) {
    size_t size = ArrSize(indices);
    dim3 blocks;
    dim3 threads;
    ThreadBlock1D(threads, blocks, size);
    const int *indices_data = (const int *)indices->data;
    const float *values_data = (const float *)values->data;
    const float *qparam_data = (const float *)qparam->data;
    assert(embedding->ndim == 2);
    size_t nrow = embedding->shape[0];
    size_t dim = embedding->shape[1];
    HetuRandomState &cudars = GetRandomState(dim);
    if (digit == 16) {
        uint16_t *embed_data = (uint16_t *)embedding->data;

        if (stream_handle) {
            assign_quantized_embedding_kernel<uint16_t>
                <<<blocks, threads, 0,
                   *(cudaStream_t *)stream_handle->handle>>>(
                    embed_data, indices_data, values_data, qparam_data, cudars,
                    stochastic, nrow, dim, size);
        } else {
            assign_quantized_embedding_kernel<uint16_t><<<blocks, threads>>>(
                embed_data, indices_data, values_data, qparam_data, cudars,
                stochastic, nrow, dim, size);
        }
    } else {
        uint8_t *embed_data = (uint8_t *)embedding->data;

        if (stream_handle) {
            assign_quantized_embedding_kernel<uint8_t>
                <<<blocks, threads, 0,
                   *(cudaStream_t *)stream_handle->handle>>>(
                    embed_data, indices_data, values_data, qparam_data, cudars,
                    stochastic, nrow, dim, size);
        } else {
            assign_quantized_embedding_kernel<uint8_t><<<blocks, threads>>>(
                embed_data, indices_data, values_data, qparam_data, cudars,
                stochastic, nrow, dim, size);
        }
    }
    return 0;
}
