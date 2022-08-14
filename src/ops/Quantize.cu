#include "gpu_runtime.h"

// TODO: use template instead of multiple functions

__global__ void quantize_kernel_8(const float *input, int8_t *output,
                                  float scale, int64_t zero_point, int64_t qmin,
                                  int64_t qmax, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    float rvalue = input[ind];
    int64_t qvalue =
        static_cast<int64_t>(nearbyint(rvalue / scale) + zero_point);
    qvalue = max(qvalue, qmin);
    qvalue = min(qvalue, qmax);
    output[ind] = qvalue;
}

__global__ void quantize_kernel_16(const float *input, int16_t *output,
                                   float scale, int64_t zero_point,
                                   int64_t qmin, int64_t qmax, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    float rvalue = input[ind];
    int64_t qvalue =
        static_cast<int64_t>(nearbyint(rvalue / scale) + zero_point);
    qvalue = max(qvalue, qmin);
    qvalue = min(qvalue, qmax);
    output[ind] = qvalue;
}

int DLGpuQuantize(const DLArrayHandle input, DLArrayHandle output, int digit,
                  float scale, int64_t zero_point,
                  DLStreamHandle stream_handle = NULL) {
    size_t size = ArrSize(input);
    float *input_data = (float *)input->data;
    dim3 blocks;
    dim3 threads;
    ThreadBlock1D(threads, blocks, size);
    // auto dtype = int8_t;
    int64_t qmin, qmax;
    if (digit == 8) {
        int8_t *output_data = (int8_t *)output->data;
        qmin = std::numeric_limits<int8_t>::min();
        qmax = std::numeric_limits<int8_t>::max();

        if (stream_handle)
            quantize_kernel_8<<<blocks, threads, 0,
                                *(cudaStream_t *)stream_handle->handle>>>(
                input_data, output_data, scale, zero_point, qmin, qmax, size);
        else
            quantize_kernel_8<<<blocks, threads>>>(
                input_data, output_data, scale, zero_point, qmin, qmax, size);
    } else if (digit == 16) {
        int16_t *output_data = (int16_t *)output->data;
        qmin = std::numeric_limits<int16_t>::min();
        qmax = std::numeric_limits<int16_t>::max();

        if (stream_handle)
            quantize_kernel_16<<<blocks, threads, 0,
                                 *(cudaStream_t *)stream_handle->handle>>>(
                input_data, output_data, scale, zero_point, qmin, qmax, size);
        else
            quantize_kernel_16<<<blocks, threads>>>(
                input_data, output_data, scale, zero_point, qmin, qmax, size);
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
        constexpr int64_t qmin = std::numeric_limits<int8_t>::min();
        constexpr int64_t qmax = std::numeric_limits<int8_t>::max();
        if (stream_handle)
            dequantize_kernel_8<<<blocks, threads, 0,
                                  *(cudaStream_t *)stream_handle->handle>>>(
                input_data, output_data, scale, zero_point, size);
        else
            dequantize_kernel_8<<<blocks, threads>>>(input_data, output_data,
                                                     scale, zero_point, size);
    } else if (digit == 16) {
        int16_t *input_data = (int16_t *)input->data;
        constexpr int64_t qmin = std::numeric_limits<int16_t>::min();
        constexpr int64_t qmax = std::numeric_limits<int16_t>::max();
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
