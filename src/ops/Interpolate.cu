#include "gpu_runtime.h"

__device__ float cubic_convolution1(const float x, const float a) {
    return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
}

__device__ float cubic_convolution2(const float x, const float a) {
    return ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a;
}

__device__ float kecubic_interp(float x0, float x1, float x2, float x3,
                                float t) {
    float coeffs[4];
    float A = -0.75;
    float y1 = t;
    float y2 = 1.0 - t;
    coeffs[0] = cubic_convolution2(y1 + 1.0, A);
    coeffs[1] = cubic_convolution1(y1, A);
    coeffs[2] = cubic_convolution1(y2, A);
    coeffs[3] = cubic_convolution2(y2 + 1.0, A);
    return x0 * coeffs[0] + x1 * coeffs[1] + x2 * coeffs[2] + x3 * coeffs[3];
}

__device__ void get_cubic_upsample_coefficients(float coeffs[4], float t) {
    float A = -0.75;
    float y1 = t;
    float y2 = 1.0 - t;
    coeffs[0] = cubic_convolution2(y1 + 1.0, A);
    coeffs[1] = cubic_convolution1(y1, A);
    coeffs[2] = cubic_convolution1(y2, A);
    coeffs[3] = cubic_convolution2(y2 + 1.0, A);
}

__global__ void bicubic_interp_kernel(const float *input, int n, int c,
                                      int in_h, int in_w, float *output,
                                      int out_h, int out_w, float ratio_h,
                                      float ratio_w, bool align_corners) {
    int nthreads = n * c * out_h * out_w;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nthreads)
        return;
    int stride = blockDim.x * gridDim.x;

    int in_hw = in_h * in_w;
    int out_hw = out_h * out_w;
    int in_chw = c * in_hw;
    int out_chw = c * out_hw;

    for (; tid < nthreads; tid += stride) {
        int n_id = tid / out_chw;
        int chw_id = tid % out_chw;
        int c_id = chw_id / out_hw;
        int hw_id = chw_id % out_hw;
        int out_h_id = hw_id / out_w;
        int out_w_id = hw_id % out_w;

        float in_h_id_ = align_corners ? (ratio_h * out_h_id) :
                                         (ratio_h * (out_h_id + 0.5) - 0.5);
        int in_h_id = floorf(in_h_id_);
        float in_h_delta = in_h_id_ - in_h_id;

        float in_w_id_ = align_corners ? (ratio_w * out_w_id) :
                                         (ratio_w * (out_w_id + 0.5) - 0.5);
        int in_w_id = floorf(in_w_id_);
        float in_w_delta = in_w_id_ - in_w_id;

        float coefficients[4];
        float x0, x1, x2, x3;

        for (int k = 0; k < 4; k++) {
            int tmp_h = max(min(in_h_id - 1 + k, in_h - 1), 0);
            int tmp_w0 = max(min(in_w_id - 1, in_w - 1), 0);
            int tmp_w1 = max(min(in_w_id + 0, in_w - 1), 0);
            int tmp_w2 = max(min(in_w_id + 1, in_w - 1), 0);
            int tmp_w3 = max(min(in_w_id + 2, in_w - 1), 0);

            x0 = input[n_id * in_chw + c_id * in_hw + tmp_h * in_w + tmp_w0];
            x1 = input[n_id * in_chw + c_id * in_hw + tmp_h * in_w + tmp_w1];
            x2 = input[n_id * in_chw + c_id * in_hw + tmp_h * in_w + tmp_w2];
            x3 = input[n_id * in_chw + c_id * in_hw + tmp_h * in_w + tmp_w3];

            coefficients[k] = kecubic_interp(x0, x1, x2, x3, in_w_delta);
        }
        float val =
            kecubic_interp(coefficients[0], coefficients[1], coefficients[2],
                           coefficients[3], in_h_delta);
        output[tid] = val;
    }
}

__global__ void bicubic_interp_gradient_kernel(
    float *input, int n, int c, int in_h, int in_w, const float *output,
    int out_h, int out_w, float ratio_h, float ratio_w, bool align_corners) {
    int nthreads = n * c * out_h * out_w;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nthreads)
        return;
    int stride = blockDim.x * gridDim.x;

    int in_hw = in_h * in_w;
    int out_hw = out_h * out_w;
    int in_chw = c * in_hw;
    int out_chw = c * out_hw;

    for (; tid < nthreads; tid += stride) {
        int n_id = tid / out_chw;
        int chw_id = tid % out_chw;
        int c_id = chw_id / out_hw;
        int hw_id = chw_id % out_hw;
        int out_h_id = hw_id / out_w;
        int out_w_id = hw_id % out_w;

        float in_h_id_ = align_corners ? (ratio_h * out_h_id) :
                                         (ratio_h * (out_h_id + 0.5) - 0.5);
        int in_h_id = floorf(in_h_id_);
        float in_h_delta = in_h_id_ - in_h_id;

        float in_w_id_ = align_corners ? (ratio_w * out_w_id) :
                                         (ratio_w * (out_w_id + 0.5) - 0.5);
        int in_w_id = floorf(in_w_id_);
        float in_w_delta = in_w_id_ - in_w_id;

        float coeffs_h[4];
        float coeffs_w[4];

        get_cubic_upsample_coefficients(coeffs_h, in_h_delta);
        get_cubic_upsample_coefficients(coeffs_w, in_w_delta);

        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                int tmp_h = max(min(in_h_id - 1 + i, in_h - 1), 0);
                int tmp_w = max(min(in_w_id - 1 + j, in_w - 1), 0);
                float addend = output[tid] * coeffs_h[i] * coeffs_w[j];
                atomicAdd(
                    &input[n_id * in_chw + c_id * in_hw + tmp_h * in_w + tmp_w],
                    addend);
            }
        }
    }
}

int DLGpuBicubicInterpolate(const DLArrayHandle input, DLArrayHandle output,
                            bool align_corners,
                            DLStreamHandle stream_handle = NULL) {
    const int kThreadsPerBlock = 1024;

    int input_N = input->shape[0];
    int input_C = input->shape[1];
    int input_H = input->shape[2];
    int input_W = input->shape[3];
    int output_H = output->shape[2];
    int output_W = output->shape[3];

    float ratio_h = 0.f;
    float ratio_w = 0.f;

    ratio_h = (align_corners) ? (float)(input_H - 1) / (output_H - 1) :
                                (float)(input_H) / output_H;

    ratio_w = (align_corners) ? (float)(input_W - 1) / (output_W - 1) :
                                (float)(input_W) / output_W;

    const float *input_data = (const float *)input->data;
    float *output_data = (float *)output->data;

    int output_size = input_N * input_C * output_H * output_W;

    int blocks = (output_size + kThreadsPerBlock - 1) / kThreadsPerBlock;
    int threads = kThreadsPerBlock;

    if (stream_handle)
        bicubic_interp_kernel<<<blocks, threads, 0,
                                *(cudaStream_t *)stream_handle->handle>>>(
            input_data, input_N, input_C, input_H, input_W, output_data,
            output_H, output_W, ratio_h, ratio_w, align_corners);

    else
        bicubic_interp_kernel<<<blocks, threads>>>(
            input_data, input_N, input_C, input_H, input_W, output_data,
            output_H, output_W, ratio_h, ratio_w, align_corners);

    return 0;
}

int DLGpuBicubicInterpolateGradient(DLArrayHandle input,
                                    const DLArrayHandle output,
                                    bool align_corners,
                                    DLStreamHandle stream_handle = NULL) {
    const int kThreadsPerBlock = 1024;

    int input_N = input->shape[0];
    int input_C = input->shape[1];
    int input_H = input->shape[2];
    int input_W = input->shape[3];
    int output_H = output->shape[2];
    int output_W = output->shape[3];

    float ratio_h = 0.f;
    float ratio_w = 0.f;

    ratio_h = (align_corners) ? (float)(input_H - 1) / (output_H - 1) :
                                (float)(input_H) / output_H;

    ratio_w = (align_corners) ? (float)(input_W - 1) / (output_W - 1) :
                                (float)(input_W) / output_W;

    float *input_data = (float *)input->data;
    const float *output_data = (const float *)output->data;

    int output_size = input_N * input_C * output_H * output_W;

    int blocks = (output_size + kThreadsPerBlock - 1) / kThreadsPerBlock;
    int threads = kThreadsPerBlock;

    if (stream_handle)
        bicubic_interp_gradient_kernel<<<
            blocks, threads, 0, *(cudaStream_t *)stream_handle->handle>>>(
            input_data, input_N, input_C, input_H, input_W, output_data,
            output_H, output_W, ratio_h, ratio_w, align_corners);

    else
        bicubic_interp_gradient_kernel<<<blocks, threads>>>(
            input_data, input_N, input_C, input_H, input_W, output_data,
            output_H, output_W, ratio_h, ratio_w, align_corners);

    return 0;
}