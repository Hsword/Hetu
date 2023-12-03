#include "gpu_runtime.h"
#include "gpu_functions.cuh"
#include "random.h"

template <class T>
__global__ void rounding_kernel(const float *input, T *output, float scale,
                                float minele, HetuRandomState cudars,
                                bool stochastic, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    float cur_value = input[ind];
    T out;
    if (stochastic) {
        out = stochastic_rounding<T>(cur_value, scale, minele, cudars, ind);
    } else {
        out = fixed_rounding<T>(cur_value, scale, minele);
    }
    output[ind] = out;
}

int DLGpuRoundingToInt(const DLArrayHandle input, DLArrayHandle output,
                       float scale, float minele, int digit, bool stochastic,
                       DLStreamHandle stream_handle = NULL) {
    size_t size = ArrSize(input);
    const float *input_data = (const float *)input->data;
    dim3 blocks;
    dim3 threads;
    ThreadBlock1D(threads, blocks, size);
    HetuRandomState &cudars = GetRandomState(1);
    if (digit == 8) {
        uint8_t *output_data = (uint8_t *)output->data;

        if (stream_handle)
            rounding_kernel<uint8_t>
                <<<blocks, threads, 0,
                   *(cudaStream_t *)stream_handle->handle>>>(
                    input_data, output_data, scale, minele, cudars, stochastic,
                    size);
        else
            rounding_kernel<uint8_t>
                <<<blocks, threads>>>(input_data, output_data, scale, minele,
                                      cudars, stochastic, size);
    } else if (digit == 16) {
        uint16_t *output_data = (uint16_t *)output->data;

        if (stream_handle)
            rounding_kernel<uint16_t>
                <<<blocks, threads, 0,
                   *(cudaStream_t *)stream_handle->handle>>>(
                    input_data, output_data, scale, minele, cudars, stochastic,
                    size);
        else
            rounding_kernel<uint16_t>
                <<<blocks, threads>>>(input_data, output_data, scale, minele,
                                      cudars, stochastic, size);
    } else {
        assert(false);
    }
    return 0;
}

template <class T>
__global__ void dequantize_kernel(const T *input, float *output, float scale,
                                  float minele, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    T qvalue = input[ind];
    float rvalue = static_cast<float>(qvalue) * scale + minele;
    output[ind] = rvalue;
}

int DLGpuDequantize(const DLArrayHandle input, DLArrayHandle output, int digit,
                    float scale, float minele,
                    DLStreamHandle stream_handle = NULL) {
    size_t size = ArrSize(output);
    float *output_data = (float *)output->data;
    dim3 blocks;
    dim3 threads;
    ThreadBlock1D(threads, blocks, size);
    if (digit == 8) {
        uint8_t *input_data = (uint8_t *)input->data;
        if (stream_handle)
            dequantize_kernel<uint8_t>
                <<<blocks, threads, 0,
                   *(cudaStream_t *)stream_handle->handle>>>(
                    input_data, output_data, scale, minele, size);
        else
            dequantize_kernel<uint8_t><<<blocks, threads>>>(
                input_data, output_data, scale, minele, size);
    } else if (digit == 16) {
        uint16_t *input_data = (uint16_t *)input->data;
        if (stream_handle)
            dequantize_kernel<uint16_t>
                <<<blocks, threads, 0,
                   *(cudaStream_t *)stream_handle->handle>>>(
                    input_data, output_data, scale, minele, size);
        else
            dequantize_kernel<uint16_t><<<blocks, threads>>>(
                input_data, output_data, scale, minele, size);
    } else {
        assert(false);
    }
    return 0;
}
