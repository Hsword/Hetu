#include "gpu_runtime.h"

__global__ void max_pool_forward_kernel(const int nthreads,
                                        const float *input_data,
                                        const int input_H, const int input_W,
                                        const int input_C, const int pooled_H,
                                        const int pooled_W, const int kernel_h,
                                        const int kernel_w, const int stride,
                                        const int padding, float *output_data) {
    const float kMinFLOAT8X4 = -FLT_MAX;

    for (int index = blockDim.x * blockIdx.x + threadIdx.x; index < nthreads;
         index += gridDim.x * blockDim.x) {
        int pw = index % pooled_W;
        int ph = (index / pooled_W) % pooled_H;
        int c = (index / pooled_W / pooled_H) % input_C;
        int n = index / pooled_W / pooled_H / input_C;

        int hstart = ph * stride - padding;
        int wstart = pw * stride - padding;
        int hend = min(hstart + kernel_h, input_H);
        int wend = min(wstart + kernel_w, input_W);

        hstart = max(hstart, 0);
        wstart = max(wstart, 0);

        float maxval = kMinFLOAT8X4;

        const float *input_data_n =
            input_data + n * input_C * input_H * input_W;

        for (int h = hstart; h < hend; h++) {
            for (int w = wstart; w < wend; w++) {
                int idx = (c * input_H + h) * input_W + w;
                if (input_data_n[idx] > maxval) {
                    maxval = input_data_n[idx];
                }
            }
        }

        output_data[index] = maxval;
    }
}

int DLGpuMax_Pooling2d(const DLArrayHandle input, const int kernel_H,
                       const int kernel_W, DLArrayHandle output,
                       const int padding, const int stride,
                       DLStreamHandle stream_handle = NULL) {
    const int kThreadsPerBlock = 1024;

    int input_N = input->shape[0];
    int input_C = input->shape[1];
    int input_H = input->shape[2];
    int input_W = input->shape[3];
    int pooled_H = output->shape[2];
    int pooled_W = output->shape[3];

    const float *input_data = (const float *)input->data;
    float *output_data = (float *)output->data;

    int output_size = input_N * input_C * pooled_H * pooled_W;

    int blocks = (output_size + kThreadsPerBlock - 1) / kThreadsPerBlock;
    int threads = kThreadsPerBlock;

    if (stream_handle)
        max_pool_forward_kernel<<<blocks, threads, 0,
                                  *(cudaStream_t *)stream_handle->handle>>>(
            output_size, input_data, input_H, input_W, input_C, pooled_H,
            pooled_W, kernel_H, kernel_W, stride, padding, output_data);
    else
        max_pool_forward_kernel<<<blocks, threads>>>(
            output_size, input_data, input_H, input_W, input_C, pooled_H,
            pooled_W, kernel_H, kernel_W, stride, padding, output_data);

    return 0;
}

__global__ void max_pool_backward_kernel(
    const int nthreads, const float *input_data, const int input_H,
    const int input_W, const int input_C, const int pooled_H,
    const int pooled_W, const int kernel_H, const int kernel_W,
    const int stride, const int padding, const float *input_grad_data,
    float *output_grad_data) {
    const float kMinFLOAT8X4 = -FLT_MAX;

    for (int index = blockDim.x * blockIdx.x + threadIdx.x; index < nthreads;
         index += gridDim.x * blockDim.x) {
        int pw = index % pooled_W;
        int ph = (index / pooled_W) % pooled_H;
        int c = (index / pooled_W / pooled_H) % input_C;
        int n = index / pooled_W / pooled_H / input_C;

        int hstart = ph * stride - padding;
        int wstart = pw * stride - padding;
        int hend = min(hstart + kernel_H, input_H);
        int wend = min(wstart + kernel_W, input_W);

        hstart = max(hstart, 0);
        wstart = max(wstart, 0);

        float maxval = kMinFLOAT8X4;
        int maxidx = -1;

        const float *input_data_n =
            input_data + n * input_C * input_H * input_W;
        float *output_grad_data_n =
            output_grad_data + n * input_C * input_H * input_W;

        for (int h = hstart; h < hend; h++) {
            for (int w = wstart; w < wend; w++) {
                int idx = (c * input_H + h) * input_W + w;
                if (input_data_n[idx] > maxval) {
                    maxidx = idx;
                    maxval = input_data_n[idx];
                }
            }
        }

        if (maxidx != -1) {
            atomicAdd(output_grad_data_n + maxidx, input_grad_data[index]);
        }
    }
}

int DLGpuMax_Pooling2d_gradient(const DLArrayHandle input,
                                const DLArrayHandle input_grad,
                                const int kernel_H, const int kernel_W,
                                DLArrayHandle output_grad, const int padding,
                                const int stride,
                                DLStreamHandle stream_handle = NULL) {
    const int kThreadsPerBlock = 1024;
    int input_N = input->shape[0];
    int input_C = input->shape[1];
    int input_H = input->shape[2];
    int input_W = input->shape[3];
    int pooled_H = input_grad->shape[2];
    int pooled_W = input_grad->shape[3];

    const float *input_data = (const float *)input->data;
    const float *input_grad_data = (const float *)input_grad->data;
    float *output_grad_data = (float *)output_grad->data;

    int input_grad_size = input_N * input_C * pooled_H * pooled_W;
    int output_grad_size = input_N * input_C * input_H * input_W;
    int blocks = (input_grad_size + kThreadsPerBlock - 1) / kThreadsPerBlock;
    int threads = kThreadsPerBlock;
    cudaMemset(output_grad_data, 0, output_grad_size * sizeof(float));
    if (stream_handle)
        max_pool_backward_kernel<<<blocks, threads, 0,
                                   *(cudaStream_t *)stream_handle->handle>>>(
            input_grad_size, input_data, input_H, input_W, input_C, pooled_H,
            pooled_W, kernel_H, kernel_W, stride, padding, input_grad_data,
            output_grad_data);
    else
        max_pool_backward_kernel<<<blocks, threads>>>(
            input_grad_size, input_data, input_H, input_W, input_C, pooled_H,
            pooled_W, kernel_H, kernel_W, stride, padding, input_grad_data,
            output_grad_data);

    return 0;
}
