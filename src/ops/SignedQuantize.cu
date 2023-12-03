#include "gpu_runtime.h"
#include "gpu_functions.cuh"
#include "random.h"

template <class T>
__global__ void
signed_rounding_kernel(const float *input, T *output, float scale, float middle,
                       HetuRandomState cudars, bool stochastic, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    float cur_value = input[ind];
    T out;
    if (stochastic) {
        out = signed_stochastic_rounding<T>(cur_value, scale, middle, cudars,
                                            ind);
    } else {
        out = signed_fixed_rounding<T>(cur_value, scale, middle);
    }
    output[ind] = out;
}

int DLGpuRoundingToSignedInt(const DLArrayHandle input, DLArrayHandle output,
                             float scale, float middle, int digit,
                             bool stochastic,
                             DLStreamHandle stream_handle = NULL) {
    size_t size = ArrSize(input);
    const float *input_data = (const float *)input->data;
    dim3 blocks;
    dim3 threads;
    ThreadBlock1D(threads, blocks, size);
    HetuRandomState &cudars = GetRandomState(1);
    if (digit == 8) {
        int8_t *output_data = (int8_t *)output->data;

        if (stream_handle)
            signed_rounding_kernel<int8_t>
                <<<blocks, threads, 0,
                   *(cudaStream_t *)stream_handle->handle>>>(
                    input_data, output_data, scale, middle, cudars, stochastic,
                    size);
        else
            signed_rounding_kernel<int8_t>
                <<<blocks, threads>>>(input_data, output_data, scale, middle,
                                      cudars, stochastic, size);
    } else if (digit == 16) {
        int16_t *output_data = (int16_t *)output->data;

        if (stream_handle)
            signed_rounding_kernel<int16_t>
                <<<blocks, threads, 0,
                   *(cudaStream_t *)stream_handle->handle>>>(
                    input_data, output_data, scale, middle, cudars, stochastic,
                    size);
        else
            signed_rounding_kernel<int16_t>
                <<<blocks, threads>>>(input_data, output_data, scale, middle,
                                      cudars, stochastic, size);
    } else {
        assert(false);
    }
    return 0;
}

template <class T>
__global__ void signed_dequantize_kernel(const T *input, float *output,
                                         float scale, float middle,
                                         size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    T qvalue = input[ind];
    float rvalue = static_cast<float>(qvalue) * scale + middle;
    output[ind] = rvalue;
}

int DLGpuDequantizeSigned(const DLArrayHandle input, DLArrayHandle output,
                          int digit, float scale, float middle,
                          DLStreamHandle stream_handle = NULL) {
    size_t size = ArrSize(output);
    float *output_data = (float *)output->data;
    dim3 blocks;
    dim3 threads;
    ThreadBlock1D(threads, blocks, size);
    if (digit == 8) {
        int8_t *input_data = (int8_t *)input->data;
        if (stream_handle)
            signed_dequantize_kernel<int8_t>
                <<<blocks, threads, 0,
                   *(cudaStream_t *)stream_handle->handle>>>(
                    input_data, output_data, scale, middle, size);
        else
            signed_dequantize_kernel<int8_t><<<blocks, threads>>>(
                input_data, output_data, scale, middle, size);
    } else if (digit == 16) {
        int16_t *input_data = (int16_t *)input->data;
        if (stream_handle)
            signed_dequantize_kernel<int16_t>
                <<<blocks, threads, 0,
                   *(cudaStream_t *)stream_handle->handle>>>(
                    input_data, output_data, scale, middle, size);
        else
            signed_dequantize_kernel<int16_t><<<blocks, threads>>>(
                input_data, output_data, scale, middle, size);
    } else {
        assert(false);
    }
    return 0;
}

template <class T>
__global__ void signed_rounding_with_scale_kernel(
    const float *input, T *output, const float *scale, float middle,
    HetuRandomState cudars, bool stochastic, size_t dim, size_t rsize) {
    size_t rind = blockIdx.x * blockDim.x + threadIdx.x;
    if (rind >= rsize)
        return;
    size_t offset = dim * rind;
    const float *cur_input = input + offset;
    float cur_scale = scale[rind];
    T *cur_out = output + offset;
    for (int i = 0; i < dim; ++i) {
        float cur_value = cur_input[i];
        T out;
        if (stochastic) {
            out = signed_stochastic_rounding<T>(cur_value, cur_scale, middle,
                                                cudars, rind);
        } else {
            out = signed_fixed_rounding<T>(cur_value, cur_scale, middle);
        }
        cur_out[i] = out;
    }
}

int DLGpuQuantizeEmbeddingWithScale(const DLArrayHandle input,
                                    const DLArrayHandle scale,
                                    DLArrayHandle output, float middle,
                                    int digit,
                                    DLStreamHandle stream_handle = NULL) {
    assert(input->ndim == 2);
    size_t rsize = input->shape[0];
    size_t dim = input->shape[1];
    const float *input_data = (const float *)input->data;
    const float *scale_data = (const float *)scale->data;
    dim3 blocks;
    dim3 threads;
    ThreadBlock1D(threads, blocks, rsize);
    HetuRandomState &cudars = GetRandomState(dim);
    if (digit == 8) {
        int8_t *output_data = (int8_t *)output->data;

        if (stream_handle)
            signed_rounding_with_scale_kernel<int8_t>
                <<<blocks, threads, 0,
                   *(cudaStream_t *)stream_handle->handle>>>(
                    input_data, output_data, scale_data, middle, cudars, true,
                    dim, rsize);
        else
            signed_rounding_with_scale_kernel<int8_t>
                <<<blocks, threads>>>(input_data, output_data, scale_data,
                                      middle, cudars, true, dim, rsize);
    } else if (digit == 16) {
        int16_t *output_data = (int16_t *)output->data;

        if (stream_handle)
            signed_rounding_with_scale_kernel<int16_t>
                <<<blocks, threads, 0,
                   *(cudaStream_t *)stream_handle->handle>>>(
                    input_data, output_data, scale_data, middle, cudars, true,
                    dim, rsize);
        else
            signed_rounding_with_scale_kernel<int16_t>
                <<<blocks, threads>>>(input_data, output_data, scale_data,
                                      middle, cudars, true, dim, rsize);
    } else {
        assert(false);
    }
    return 0;
}

template <class T>
__global__ void dequantize_lookup_with_scale_kernel(
    const T *input, const int *indices, const float *scale, float *output,
    float middle, size_t nrow, size_t dim, size_t size) {
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
        float cur_scale = scale[id];
        for (int i = 0; i < dim; i++) {
            output_ptr[i] = (float)input_ptr[i] * cur_scale + middle;
        }
    }
}

int DLGpuQuantizedEmbeddingLookupWithScale(
    const DLArrayHandle input, const DLArrayHandle indices,
    const DLArrayHandle scale, DLArrayHandle output, int digit, float middle,
    DLStreamHandle stream_handle = NULL) {
    assert(input->ndim == 2);
    size_t size = ArrSize(indices);
    size_t nrow = input->shape[0];
    size_t dim = input->shape[1];
    const int *indices_data = (const int *)indices->data;
    const float *scale_data = (const float *)scale->data;
    float *output_data = (float *)output->data;
    dim3 blocks;
    dim3 threads;
    ThreadBlock1D(threads, blocks, size);
    if (digit == 8) {
        int8_t *input_data = (int8_t *)input->data;

        if (stream_handle)
            dequantize_lookup_with_scale_kernel<int8_t>
                <<<blocks, threads, 0,
                   *(cudaStream_t *)stream_handle->handle>>>(
                    input_data, indices_data, scale_data, output_data, middle,
                    nrow, dim, size);
        else
            dequantize_lookup_with_scale_kernel<int8_t>
                <<<blocks, threads>>>(input_data, indices_data, scale_data,
                                      output_data, middle, nrow, dim, size);
    } else if (digit == 16) {
        int16_t *input_data = (int16_t *)input->data;

        if (stream_handle)
            dequantize_lookup_with_scale_kernel<int16_t>
                <<<blocks, threads, 0,
                   *(cudaStream_t *)stream_handle->handle>>>(
                    input_data, indices_data, scale_data, output_data, middle,
                    nrow, dim, size);
        else
            dequantize_lookup_with_scale_kernel<int16_t>
                <<<blocks, threads>>>(input_data, indices_data, scale_data,
                                      output_data, middle, nrow, dim, size);

    } else {
        assert(false);
    }
    return 0;
}

template <class T>
__global__ void lsq_rounding_kernel(const float *input, const float *scale,
                                    float *output, float middle, size_t dim,
                                    size_t size) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size)
        return;

    float cur_scale = scale[index / dim];
    float value = input[index];
    float result;
    float max_limit = __signed_numeric_max_on_device<T>();
    float min_limit = __signed_numeric_min_on_device<T>();
    if (value >= max_limit) {
        result = max_limit;
    } else if (value <= min_limit) {
        result = min_limit;
    } else {
        result = floorf(value + 0.5);
    }
    output[index] = result * cur_scale + middle;
}

int DLGpuLSQRounding(const DLArrayHandle input, const DLArrayHandle scale,
                     DLArrayHandle output, int digit, float middle,
                     DLStreamHandle stream_handle = NULL) {
    size_t size = ArrSize(input);
    size_t dim = input->shape[input->ndim - 1];
    const float *input_data = (const float *)input->data;
    const float *scale_data = (const float *)scale->data;
    float *output_data = (float *)output->data;
    dim3 blocks;
    dim3 threads;
    ThreadBlock1D(threads, blocks, size);
    if (digit == 8) {
        if (stream_handle)
            lsq_rounding_kernel<int8_t>
                <<<blocks, threads, 0,
                   *(cudaStream_t *)stream_handle->handle>>>(
                    input_data, scale_data, output_data, middle, dim, size);
        else
            lsq_rounding_kernel<int8_t><<<blocks, threads>>>(
                input_data, scale_data, output_data, middle, dim, size);

    } else if (digit == 16) {
        if (stream_handle)
            lsq_rounding_kernel<int16_t>
                <<<blocks, threads, 0,
                   *(cudaStream_t *)stream_handle->handle>>>(
                    input_data, scale_data, output_data, middle, dim, size);

        else
            lsq_rounding_kernel<int16_t><<<blocks, threads>>>(
                input_data, scale_data, output_data, middle, dim, size);

    } else {
        assert(false);
    }
    return 0;
}

template <class T>
__global__ void lsq_rounding_gradient_kernel(const float *input, float *output,
                                             size_t size) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size)
        return;

    float value = input[index];
    float result;
    float max_limit = __signed_numeric_max_on_device<T>();
    float min_limit = __signed_numeric_min_on_device<T>();
    if (value >= max_limit) {
        result = max_limit;
    } else if (value <= min_limit) {
        result = min_limit;
    } else {
        result = floorf(value + 0.5) - value;
    }
    output[index] = result;
}

int DLGpuLSQRoundingGradient(const DLArrayHandle input, DLArrayHandle output,
                             int digit, DLStreamHandle stream_handle = NULL) {
    size_t size = ArrSize(input);
    const float *input_data = (const float *)input->data;
    float *output_data = (float *)output->data;
    dim3 blocks;
    dim3 threads;
    ThreadBlock1D(threads, blocks, size);
    if (digit == 8) {
        if (stream_handle)
            lsq_rounding_gradient_kernel<int8_t>
                <<<blocks, threads, 0,
                   *(cudaStream_t *)stream_handle->handle>>>(input_data,
                                                             output_data, size);
        else
            lsq_rounding_gradient_kernel<int8_t>
                <<<blocks, threads>>>(input_data, output_data, size);

    } else if (digit == 16) {
        if (stream_handle)
            lsq_rounding_gradient_kernel<int16_t>
                <<<blocks, threads, 0,
                   *(cudaStream_t *)stream_handle->handle>>>(input_data,
                                                             output_data, size);
        else
            lsq_rounding_gradient_kernel<int16_t>
                <<<blocks, threads>>>(input_data, output_data, size);
    } else {
        assert(false);
    }
    return 0;
}
