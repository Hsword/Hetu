#include "gpu_runtime.h"

// TODO: use template instead of multiple functions

__global__ void prepack_kernel_8(const float *input, uint8_t *output,
                                 float *qparams, size_t dim, size_t rsize) {
    size_t rind = blockIdx.x * blockDim.x + threadIdx.x;
    if (rind >= rsize)
        return;
    const float kEpsilon = 1e-8f;
    size_t offset = dim * rind;
    const float *cur_input = input + offset;
    float *cur_qparam = qparams + 2 * rind;
    uint8_t *cur_out = output + offset;
    float maxele = cur_input[0];
    float minele = maxele;
    for (int i = 1; i < dim; ++i) {
        float cur_val = cur_input[i];
        maxele = max(maxele, cur_val);
        minele = min(minele, cur_val);
    }
    float range = maxele - minele;
    cur_qparam[0] = range / 255.0f;
    cur_qparam[1] = minele;
    float inverse = 255.0f / (range + kEpsilon);
    for (int i = 0; i < dim; ++i) {
        cur_out[i] = lrint((cur_input[i] - minele) * inverse);
    }
}

__global__ void prepack_kernel_16(const float *input, uint16_t *output,
                                  float *qparams, size_t dim, size_t rsize) {
    size_t rind = blockIdx.x * blockDim.x + threadIdx.x;
    if (rind >= rsize)
        return;
    const float kEpsilon = 1e-8f;
    size_t offset = dim * rind;
    const float *cur_input = input + offset;
    float *cur_qparam = qparams + 2 * rind;
    uint16_t *cur_out = output + offset;
    float maxele = cur_input[0];
    float minele = maxele;
    for (int i = 1; i < dim; ++i) {
        float cur_val = cur_input[i];
        maxele = max(maxele, cur_val);
        minele = min(minele, cur_val);
    }
    float range = maxele - minele;
    cur_qparam[0] = range / 65535.0f;
    cur_qparam[1] = minele;
    float inverse = 65535.0f / (range + kEpsilon);
    for (int i = 0; i < dim; ++i) {
        cur_out[i] = lrint((cur_input[i] - minele) * inverse);
    }
}

__global__ void quantized_embedding_lookup_kernel_8(const uint8_t *input,
                                                    const int *indices,
                                                    float *output,
                                                    float *qparams, size_t dim,
                                                    size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    int cur_ind = indices[ind];
    const uint8_t *cur_input = input + dim * cur_ind;
    float *cur_qpar = qparams + 2 * cur_ind;
    float *cur_output = output + dim * ind;
    float scale = cur_qpar[0];
    float zero_point = cur_qpar[1];
    for (int i = 0; i < dim; ++i) {
        cur_output[i] = float(cur_input[i]) * scale + zero_point;
    }
}

__global__ void quantized_embedding_lookup_kernel_16(const uint16_t *input,
                                                     const int *indices,
                                                     float *output,
                                                     float *qparams, size_t dim,
                                                     size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    int cur_ind = indices[ind];
    const uint16_t *cur_input = input + dim * cur_ind;
    float *cur_qpar = qparams + 2 * cur_ind;
    float *cur_output = output + dim * ind;
    float scale = cur_qpar[0];
    float zero_point = cur_qpar[1];
    for (int i = 0; i < dim; ++i) {
        cur_output[i] = float(cur_input[i]) * scale + zero_point;
    }
}

__global__ void update_quantized_embedding_kernel_8(const float *input,
                                                    const int *indices,
                                                    uint8_t *output,
                                                    float *qparams, size_t dim,
                                                    size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    int cur_ind = indices[ind];
    const float kEpsilon = 1e-8f;
    const float *cur_input = input + dim * ind;
    float *cur_qparam = qparams + 2 * cur_ind;
    uint8_t *cur_out = output + dim * cur_ind;
    float maxele = cur_input[0];
    float minele = maxele;
    for (int i = 1; i < dim; ++i) {
        float cur_val = cur_input[i];
        maxele = max(maxele, cur_val);
        minele = min(minele, cur_val);
    }
    float range = maxele - minele;
    cur_qparam[0] = range / 255.0f;
    cur_qparam[1] = minele;
    float inverse = 255.0f / (range + kEpsilon);
    for (int i = 0; i < dim; ++i) {
        cur_out[i] = lrint((cur_input[i] - minele) * inverse);
    }
}

__global__ void update_quantized_embedding_kernel_16(const float *input,
                                                     const int *indices,
                                                     uint16_t *output,
                                                     float *qparams, size_t dim,
                                                     size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    int cur_ind = indices[ind];
    const float kEpsilon = 1e-8f;
    const float *cur_input = input + dim * ind;
    float *cur_qparam = qparams + 2 * cur_ind;
    uint16_t *cur_out = output + dim * cur_ind;
    float maxele = cur_input[0];
    float minele = maxele;
    for (int i = 1; i < dim; ++i) {
        float cur_val = cur_input[i];
        maxele = max(maxele, cur_val);
        minele = min(minele, cur_val);
    }
    float range = maxele - minele;
    cur_qparam[0] = range / 65535.0f;
    cur_qparam[1] = minele;
    float inverse = 65535.0f / (range + kEpsilon);
    for (int i = 0; i < dim; ++i) {
        cur_out[i] = lrint((cur_input[i] - minele) * inverse);
    }
}

int DLGpuPrepackEmbedding(const DLArrayHandle input, DLArrayHandle output,
                          DLArrayHandle qparams, int digit,
                          DLStreamHandle stream_handle = NULL) {
    assert(input->ndim == 2);
    size_t rsize = input->shape[0];
    size_t dim = input->shape[1];
    const float *input_data = (const float *)input->data;
    float *qparam_data = (float *)qparams->data;
    dim3 blocks;
    dim3 threads;
    ThreadBlock1D(threads, blocks, rsize);
    if (digit == 8) {
        uint8_t *output_data = (uint8_t *)output->data;

        if (stream_handle)
            prepack_kernel_8<<<blocks, threads, 0,
                               *(cudaStream_t *)stream_handle->handle>>>(
                input_data, output_data, qparam_data, dim, rsize);
        else
            prepack_kernel_8<<<blocks, threads>>>(input_data, output_data,
                                                  qparam_data, dim, rsize);
    } else if (digit == 16) {
        uint16_t *output_data = (uint16_t *)output->data;

        if (stream_handle)
            prepack_kernel_16<<<blocks, threads, 0,
                                *(cudaStream_t *)stream_handle->handle>>>(
                input_data, output_data, qparam_data, dim, rsize);
        else
            prepack_kernel_16<<<blocks, threads>>>(input_data, output_data,
                                                   qparam_data, dim, rsize);
    } else {
        assert(false);
    }
    return 0;
}

int DLGpuQuantizedEmbeddingLookup(const DLArrayHandle input,
                                  const DLArrayHandle indices,
                                  DLArrayHandle output, DLArrayHandle qparams,
                                  int digit,
                                  DLStreamHandle stream_handle = NULL) {
    assert(input->ndim == 2);
    size_t size = ArrSize(indices);
    size_t dim = input->shape[1];
    const int *indices_data = (const int *)indices->data;
    float *qparam_data = (float *)qparams->data;
    float *output_data = (float *)output->data;
    dim3 blocks;
    dim3 threads;
    ThreadBlock1D(threads, blocks, size);
    if (digit == 8) {
        uint8_t *input_data = (uint8_t *)input->data;

        if (stream_handle)
            quantized_embedding_lookup_kernel_8<<<
                blocks, threads, 0, *(cudaStream_t *)stream_handle->handle>>>(
                input_data, indices_data, output_data, qparam_data, dim, size);
        else
            quantized_embedding_lookup_kernel_8<<<blocks, threads>>>(
                input_data, indices_data, output_data, qparam_data, dim, size);
    } else if (digit == 16) {
        uint16_t *input_data = (uint16_t *)input->data;

        if (stream_handle)
            quantized_embedding_lookup_kernel_16<<<
                blocks, threads, 0, *(cudaStream_t *)stream_handle->handle>>>(
                input_data, indices_data, output_data, qparam_data, dim, size);
        else
            quantized_embedding_lookup_kernel_16<<<blocks, threads>>>(
                input_data, indices_data, output_data, qparam_data, dim, size);

    } else {
        assert(false);
    }
    return 0;
}

int DLGpuUpdateQuantizedEmbedding(const DLArrayHandle grad,
                                  const DLArrayHandle indices,
                                  DLArrayHandle embed, DLArrayHandle qparams,
                                  int digit,
                                  DLStreamHandle stream_handle = NULL) {
    assert(embed->ndim == 2);
    size_t size = ArrSize(indices);
    size_t dim = embed->shape[1];
    const float *grad_data = (const float *)grad->data;
    const int *indices_data = (const int *)indices->data;
    float *qparam_data = (float *)qparams->data;
    dim3 blocks;
    dim3 threads;
    ThreadBlock1D(threads, blocks, size);
    if (digit == 8) {
        uint8_t *embed_data = (uint8_t *)embed->data;
        if (stream_handle)
            update_quantized_embedding_kernel_8<<<
                blocks, threads, 0, *(cudaStream_t *)stream_handle->handle>>>(
                grad_data, indices_data, embed_data, qparam_data, dim, size);
        else
            update_quantized_embedding_kernel_8<<<blocks, threads>>>(
                grad_data, indices_data, embed_data, qparam_data, dim, size);
    } else if (digit == 16) {
        uint16_t *embed_data = (uint16_t *)embed->data;
        if (stream_handle)
            update_quantized_embedding_kernel_16<<<
                blocks, threads, 0, *(cudaStream_t *)stream_handle->handle>>>(
                grad_data, indices_data, embed_data, qparam_data, dim, size);
        else
            update_quantized_embedding_kernel_16<<<blocks, threads>>>(
                grad_data, indices_data, embed_data, qparam_data, dim, size);
    } else {
        assert(false);
    }
    return 0;
}

template <class T>
__global__ void dequantize_lookup_kernel(const T *input, const int *indices,
                                         float *output, float scale,
                                         float minele, size_t nrow, size_t dim,
                                         size_t size) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size)
        return;

    int id = indices[index];
    float *output_ptr = output + dim * index;
    if (id < 0 || id >= nrow) {
        for (int i = 0; i < dim; i++)
            output_ptr[i] = 0;
    } else {
        const T *input_ptr = input + dim * id;
        for (int i = 0; i < dim; i++) {
            output_ptr[i] = (float)input_ptr[i] * scale + minele;
        }
    }
}

int DLGpuUnifiedQuantizedEmbeddingLookup(const DLArrayHandle input,
                                         const DLArrayHandle indices,
                                         DLArrayHandle output, int digit,
                                         float scale, float minele,
                                         DLStreamHandle stream_handle = NULL) {
    assert(input->ndim == 2);
    size_t size = ArrSize(indices);
    size_t nrow = input->shape[0];
    size_t dim = input->shape[1];
    const int *indices_data = (const int *)indices->data;
    float *output_data = (float *)output->data;
    dim3 blocks;
    dim3 threads;
    ThreadBlock1D(threads, blocks, size);
    if (digit == 8) {
        uint8_t *input_data = (uint8_t *)input->data;

        if (stream_handle)
            dequantize_lookup_kernel<<<
                blocks, threads, 0, *(cudaStream_t *)stream_handle->handle>>>(
                input_data, indices_data, output_data, scale, minele, nrow, dim,
                size);
        else
            dequantize_lookup_kernel<<<blocks, threads>>>(
                input_data, indices_data, output_data, scale, minele, nrow, dim,
                size);
    } else if (digit == 16) {
        uint16_t *input_data = (uint16_t *)input->data;

        if (stream_handle)
            dequantize_lookup_kernel<<<
                blocks, threads, 0, *(cudaStream_t *)stream_handle->handle>>>(
                input_data, indices_data, output_data, scale, minele, nrow, dim,
                size);
        else
            dequantize_lookup_kernel<<<blocks, threads>>>(
                input_data, indices_data, output_data, scale, minele, nrow, dim,
                size);

    } else {
        assert(false);
    }
    return 0;
}
