#include "gpu_runtime.h"

__global__ void average_pooling2d(const size_t threads, const float *input_data,
                                  float *output_data, const size_t N,
                                  const size_t C, const size_t H,
                                  const size_t W, const size_t kernel_H,
                                  const size_t kernel_W, const size_t p_H,
                                  const size_t p_W, const size_t padding,
                                  const size_t stride) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= threads)
        return;
    size_t idx = id;
    size_t idx_W = idx % p_W;
    idx /= p_W;
    size_t idx_H = idx % p_H;
    idx /= p_H;
    size_t idx_C = idx % C;
    size_t idx_N = idx / C;
    int hs = (int)idx_H * stride - padding;
    int ws = (int)idx_W * stride - padding;
    size_t hend = min(hs + kernel_H, H);
    size_t wend = min(ws + kernel_W, W);
    hs = max(hs, 0);
    ws = max(ws, 0);
    float temp = 0;
    for (index_t i = hs; i < hend; i++) {
        for (index_t j = ws; j < wend; j++) {
            temp += input_data[idx_N * C * H * W + idx_C * H * W + i * W + j];
        }
    }
    output_data[id] = temp / (kernel_H * kernel_W);
}

int DLGpuAvgerage_Pooling2d(const DLArrayHandle input, const size_t kernel_H,
                            const size_t kernel_W, DLArrayHandle output,
                            const size_t padding, const size_t stride,
                            DLStreamHandle stream_handle = NULL) {
    size_t input_N = input->shape[0];
    size_t input_C = input->shape[1];
    size_t input_H = input->shape[2];
    size_t input_W = input->shape[3];
    size_t output_H = output->shape[2];
    size_t output_W = output->shape[3];
    const float *input_data = (const float *)input->data;
    float *output_data = (float *)output->data;
    size_t pooled_H = (input_H + 2 * padding - kernel_H) / stride + 1;
    size_t pooled_W = (input_W + 2 * padding - kernel_W) / stride + 1;

    size_t output_size = input_N * input_C * output_H * output_W;
    size_t BLOCKS = (output_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (stream_handle)
        average_pooling2d<<<BLOCKS, THREADS_PER_BLOCK, 0,
                            *(cudaStream_t *)stream_handle->handle>>>(
            output_size, input_data, output_data, input_N, input_C, input_H,
            input_W, kernel_H, kernel_W, pooled_H, pooled_W, padding, stride);
    else
        average_pooling2d<<<BLOCKS, THREADS_PER_BLOCK>>>(
            output_size, input_data, output_data, input_N, input_C, input_H,
            input_W, kernel_H, kernel_W, pooled_H, pooled_W, padding, stride);
    return 0;
}

__global__ void average_pooling2d_gradient(
    const size_t threads, const float *input_data, float *output_data,
    const size_t N, const size_t C, const size_t H, const size_t W,
    const size_t kernel_H, const size_t kernel_W, const size_t p_H,
    const size_t p_W, const size_t padding, const size_t stride) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= threads)
        return;
    size_t idx = id;
    size_t idx_W = idx % p_W;
    idx /= p_W;
    size_t idx_H = idx % p_H;
    idx /= p_H;
    size_t idx_C = idx % C;
    size_t idx_N = idx / C;
    size_t hs = (idx_H < kernel_H) ? 0 : (idx_H - kernel_H) / stride + 1;
    size_t hend = min(idx_H / stride + 1, H);
    size_t ws = (idx_W < kernel_W) ? 0 : (idx_W - kernel_W) / stride + 1;
    size_t wend = min(idx_W / stride + 1, W);
    float temp = 0;
    const size_t pooling_size = kernel_H * kernel_W;
    for (index_t i = hs; i < hend; i++) {
        for (index_t j = ws; j < wend; j++) {
            temp += input_data[idx_N * C * H * W + idx_C * H * W + i * W + j];
        }
    }
    output_data[id] = temp / pooling_size;
}

int DLGpuAvgerage_Pooling2d_gradient(const DLArrayHandle gradient_Y,
                                     const size_t kernel_H,
                                     const size_t kernel_W,
                                     DLArrayHandle gradient_X,
                                     const size_t padding, const size_t stride,
                                     DLStreamHandle stream_handle = NULL) {
    size_t N = gradient_Y->shape[0];
    size_t C = gradient_Y->shape[1];
    size_t H = gradient_Y->shape[2];
    size_t W = gradient_Y->shape[3];

    size_t pooled_H = gradient_X->shape[2];
    size_t pooled_W = gradient_X->shape[3];

    const float *input_data = (const float *)gradient_Y->data;
    float *output_data = (float *)gradient_X->data;

    size_t output_size = N * C * pooled_H * pooled_W;
    size_t BLOCKS = (output_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (stream_handle)
        average_pooling2d_gradient<<<BLOCKS, THREADS_PER_BLOCK, 0,
                                     *(cudaStream_t *)stream_handle->handle>>>(
            output_size, input_data, output_data, N, C, H, W, kernel_H,
            kernel_W, pooled_H, pooled_W, padding, stride);
    else
        average_pooling2d_gradient<<<BLOCKS, THREADS_PER_BLOCK>>>(
            output_size, input_data, output_data, N, C, H, W, kernel_H,
            kernel_W, pooled_H, pooled_W, padding, stride);
    return 0;
}
