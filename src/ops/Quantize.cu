#include "gpu_runtime.h"
#include "gpu_functions.cuh"

// TODO: use template instead of multiple functions

template <class T>
__global__ void rounding_kernel(const float *input, T *output, float scale,
                                float minele, unsigned long long seed,
                                bool stochastic, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    float cur_value = input[ind];
    T out;
    if (stochastic) {
        out = stochastic_rounding<T>(cur_value, scale, minele, seed, ind);
    } else {
        out = fixed_rounding<T>(cur_value, scale, minele);
    }
    output[ind] = out;
}

int DLGpuRoundingToInt(const DLArrayHandle input, DLArrayHandle output,
                       float scale, float minele, int digit,
                       unsigned long long seed, bool stochastic,
                       DLStreamHandle stream_handle = NULL) {
    size_t size = ArrSize(input);
    const float *input_data = (const float *)input->data;
    dim3 blocks;
    dim3 threads;
    ThreadBlock1D(threads, blocks, size);
    if (digit == 8) {
        uint8_t *output_data = (uint8_t *)output->data;

        if (stream_handle)
            rounding_kernel<uint8_t><<<
                blocks, threads, 0, *(cudaStream_t *)stream_handle->handle>>>(
                input_data, output_data, scale, minele, seed, stochastic, size);
        else
            rounding_kernel<uint8_t><<<blocks, threads>>>(
                input_data, output_data, scale, minele, seed, stochastic, size);
    } else if (digit == 16) {
        uint16_t *output_data = (uint16_t *)output->data;

        if (stream_handle)
            rounding_kernel<uint16_t><<<
                blocks, threads, 0, *(cudaStream_t *)stream_handle->handle>>>(
                input_data, output_data, scale, minele, seed, stochastic, size);
        else
            rounding_kernel<uint16_t><<<blocks, threads>>>(
                input_data, output_data, scale, minele, seed, stochastic, size);
    } else {
        assert(false);
    }
    return 0;
}

__global__ void dequantize_kernel_8(const int8_t *input, float *output,
                                    float scale, int64_t zero_point,
                                    size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    int8_t qvalue = input[ind];
    float rvalue = (static_cast<float>(qvalue) - zero_point) * scale;
    output[ind] = rvalue;
}

__global__ void dequantize_kernel_16(const int16_t *input, float *output,
                                     float scale, int64_t zero_point,
                                     size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    int16_t qvalue = input[ind];
    float rvalue = (static_cast<float>(qvalue) - zero_point) * scale;
    output[ind] = rvalue;
}

int DLGpuDequantize(const DLArrayHandle input, DLArrayHandle output, int digit,
                    float scale, int64_t zero_point,
                    DLStreamHandle stream_handle = NULL) {
    size_t size = ArrSize(output);
    float *output_data = (float *)output->data;
    dim3 blocks;
    dim3 threads;
    ThreadBlock1D(threads, blocks, size);
    if (digit == 8) {
        int8_t *input_data = (int8_t *)input->data;
        if (stream_handle)
            dequantize_kernel_8<<<blocks, threads, 0,
                                  *(cudaStream_t *)stream_handle->handle>>>(
                input_data, output_data, scale, zero_point, size);
        else
            dequantize_kernel_8<<<blocks, threads>>>(input_data, output_data,
                                                     scale, zero_point, size);
    } else if (digit == 16) {
        int16_t *input_data = (int16_t *)input->data;
        if (stream_handle)
            dequantize_kernel_16<<<blocks, threads, 0,
                                   *(cudaStream_t *)stream_handle->handle>>>(
                input_data, output_data, scale, zero_point, size);
        else
            dequantize_kernel_16<<<blocks, threads>>>(input_data, output_data,
                                                      scale, zero_point, size);
    } else {
        assert(false);
    }
    return 0;
}
