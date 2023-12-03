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

__global__ void reorder_into_lookup_kernel(const float *dedupemb, float *lookup,
                                           const int *id_length,
                                           const int *id_offset,
                                           const int *punique_size,
                                           size_t width, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    size_t wid = ind % width;
    ind /= width;
    int unique_size = *punique_size;
    if (ind < unique_size) {
        int l = id_length[ind], r = id_length[ind + 1];
        const float in_value = dedupemb[ind * width + wid];
        for (int i = l; i < r; ++i) {
            lookup[id_offset[i] * width + wid] = in_value;
        }
    }
}

int DLGPUReorderIntoLookup(const DLArrayHandle idoffsets,
                           const DLArrayHandle dedupemb, DLArrayHandle lookup,
                           DLStreamHandle stream_handle = NULL) {
    size_t ind_size = 1;
    for (int i = 0; i < lookup->ndim - 1; ++i) {
        ind_size *= lookup->shape[i];
    }
    size_t width = lookup->shape[lookup->ndim - 1];
    size_t size = ind_size * width;
    const int *punique_size = (const int *)idoffsets->data;
    const int *id_offset = punique_size + 1;
    const int *id_length = id_offset + ind_size;
    const float *dedupemb_data = (const float *)dedupemb->data;
    float *lookup_data = (float *)lookup->data;
    dim3 threads, blocks;
    ThreadBlock1D(threads, blocks, size);
    if (stream_handle) {
        reorder_into_lookup_kernel<<<blocks, threads, 0,
                                     *(cudaStream_t *)stream_handle->handle>>>(
            dedupemb_data, lookup_data, id_length, id_offset, punique_size,
            width, size);
    } else {
        reorder_into_lookup_kernel<<<blocks, threads>>>(
            dedupemb_data, lookup_data, id_length, id_offset, punique_size,
            width, size);
    }
    return 0;
}

template <class T>
__global__ void assign_alpt_embedding_kernel(T *embedding, const int *indices,
                                             const float *values,
                                             const float *scales, float middle,
                                             HetuRandomState cudars,
                                             bool stochastic, size_t nrow,
                                             size_t dim, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    int index = indices[ind];
    if (index >= 0 && index < nrow) {
        T *tar_ptr = embedding + index * dim;
        const float *src_ptr = values + ind * dim;
        float scale = scales[ind];
        for (size_t i = 0; i < dim; ++i) {
            float cur_value = src_ptr[i];
            T out;
            if (stochastic) {
                out = signed_stochastic_rounding<T>(cur_value, scale, middle,
                                                    cudars, ind);
            } else {
                out = signed_fixed_rounding<T>(cur_value, scale, middle);
            }
            tar_ptr[i] = out;
        }
    }
}

int DLGPUAssignALPTEmbedding(DLArrayHandle embedding,
                             const DLArrayHandle indices,
                             const DLArrayHandle values,
                             const DLArrayHandle scale, float middle, int digit,
                             bool stochastic,
                             DLStreamHandle stream_handle = NULL) {
    size_t size = ArrSize(indices);
    dim3 blocks;
    dim3 threads;
    ThreadBlock1D(threads, blocks, size);
    const int *indices_data = (const int *)indices->data;
    const float *values_data = (const float *)values->data;
    const float *scale_data = (const float *)scale->data;
    assert(embedding->ndim == 2);
    size_t nrow = embedding->shape[0];
    size_t dim = embedding->shape[1];
    HetuRandomState &cudars = GetRandomState(dim);
    if (digit == 16) {
        int16_t *embed_data = (int16_t *)embedding->data;

        if (stream_handle) {
            assign_alpt_embedding_kernel<int16_t>
                <<<blocks, threads, 0,
                   *(cudaStream_t *)stream_handle->handle>>>(
                    embed_data, indices_data, values_data, scale_data, middle,
                    cudars, stochastic, nrow, dim, size);
        } else {
            assign_alpt_embedding_kernel<int16_t><<<blocks, threads>>>(
                embed_data, indices_data, values_data, scale_data, middle,
                cudars, stochastic, nrow, dim, size);
        }
    } else {
        int8_t *embed_data = (int8_t *)embedding->data;

        if (stream_handle) {
            assign_alpt_embedding_kernel<int8_t>
                <<<blocks, threads, 0,
                   *(cudaStream_t *)stream_handle->handle>>>(
                    embed_data, indices_data, values_data, scale_data, middle,
                    cudars, stochastic, nrow, dim, size);
        } else {
            assign_alpt_embedding_kernel<int8_t><<<blocks, threads>>>(
                embed_data, indices_data, values_data, scale_data, middle,
                cudars, stochastic, nrow, dim, size);
        }
    }
    return 0;
}
