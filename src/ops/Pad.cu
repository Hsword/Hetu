#include "gpu_runtime.h"

__global__ void pad_constant_kernel(const float *input_data, float *output_data,
                                    size_t begin_N, size_t end_N, size_t N,
                                    size_t begin_C, size_t end_C, size_t C,
                                    size_t begin_H, size_t end_H, size_t H,
                                    size_t begin_W, size_t end_W, size_t W,
                                    float constant_value) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C * H * W)
        return;
    size_t idx_N = idx / (C * H * W);
    size_t idx_C = idx % (C * H * W) / (H * W);
    size_t idx_H = idx % (H * W) / W;
    size_t idx_W = idx % W;
    if (idx_N >= begin_N && idx_N < end_N && idx_C >= begin_C && idx_C < end_C
        && idx_H >= begin_H && idx_H < end_H && idx_W >= begin_W
        && idx_W < end_W) {
        output_data[idx] = input_data[(((idx_N - begin_N) * (end_C - begin_C)
                                        + idx_C - begin_C)
                                           * (end_H - begin_H)
                                       + idx_H - begin_H)
                                          * (end_W - begin_W)
                                      + idx_W - begin_W];
    } else {
        output_data[idx] = constant_value;
    }
}

// mode = 0    CONSTANT
//        1    REFLECT
//        2    SYMMETRIC
int DLGpuPad(const DLArrayHandle input_X, DLArrayHandle output_Y, int *paddings,
             int pad_len, size_t mode = 0, float constant_values = 0,
             DLStreamHandle stream_handle = NULL) {
    int len = pad_len;
    int endpoint[8];
    for (index_t i = 0; i < 4; i++) {
        if (i < (4 - len / 2)) {
            assert((input_X->shape[i]) == (output_Y->shape[i]));
            endpoint[i * 2] = input_X->shape[i] - 1;
            endpoint[i * 2 + 1] = endpoint[i * 2] + 1;
        } else {
            assert((input_X->shape[i] + paddings[(i - (4 - len / 2)) * 2]
                    + paddings[(i - (4 - len / 2)) * 2 + 1])
                   == (output_Y->shape[i]));
            endpoint[i * 2] = paddings[(i - (4 - len / 2)) * 2];
            endpoint[i * 2 + 1] =
                paddings[(i - (4 - len / 2)) * 2] + input_X->shape[i];
        }
    }
    size_t output_size = 1;
    for (index_t i = 0; i < 4; i++) {
        output_size *= (output_Y->shape[i]);
    }

    size_t blocks = (output_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (mode == 0) {
        if (stream_handle)
            pad_constant_kernel<<<blocks, THREADS_PER_BLOCK, 0,
                                  *(cudaStream_t *)stream_handle->handle>>>(
                (const float *)input_X->data, (float *)output_Y->data,
                endpoint[0], endpoint[1], output_Y->shape[0], endpoint[2],
                endpoint[3], output_Y->shape[1], endpoint[4], endpoint[5],
                output_Y->shape[2], endpoint[6], endpoint[7],
                output_Y->shape[3], constant_values);
        else
            pad_constant_kernel<<<blocks, THREADS_PER_BLOCK>>>(
                (const float *)input_X->data, (float *)output_Y->data,
                endpoint[0], endpoint[1], output_Y->shape[0], endpoint[2],
                endpoint[3], output_Y->shape[1], endpoint[4], endpoint[5],
                output_Y->shape[2], endpoint[6], endpoint[7],
                output_Y->shape[3], constant_values);
    }
    return 0;
}
__global__ void pad_constant_gradient_kernel(const float *output_grad_data,
                                             float *input_grad_data, int N,
                                             int C, int H, int W, int begin_N,
                                             int begin_C, int begin_H,
                                             int begin_W, int out_N, int out_C,
                                             int out_H, int out_W) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C * H * W)
        return;
    size_t idx_N = idx / (C * H * W);
    size_t idx_C = idx % (C * H * W) / (H * W);
    size_t idx_H = idx % (H * W) / W;
    size_t idx_W = idx % W;
    input_grad_data[idx] =
        output_grad_data[((((idx_N + begin_N) * out_C + idx_C + begin_C) * out_H
                           + idx_H + begin_H)
                              * out_W
                          + idx_W + begin_W)];
}
int DLGpuPad_gradient(const DLArrayHandle output_gradient_Y,
                      DLArrayHandle input_gradient_X, int *paddings,
                      int pad_len, size_t mode,
                      DLStreamHandle stream_handle = NULL) {
    int len = pad_len;
    int begin_p[4];
    int N = input_gradient_X->shape[0];
    int C = input_gradient_X->shape[1];
    int H = input_gradient_X->shape[2];
    int W = input_gradient_X->shape[3];

    int out_N = output_gradient_Y->shape[0];
    int out_C = output_gradient_Y->shape[1];
    int out_H = output_gradient_Y->shape[2];
    int out_W = output_gradient_Y->shape[3];

    for (int i = 0; i < 4; i++) {
        if (i < (4 - len / 2)) {
            begin_p[i] = 0;
        } else {
            begin_p[i] = paddings[(i - (4 - len / 2)) * 2];
        }
    }
    size_t input_size = 1;
    for (index_t i = 0; i < 4; i++) {
        input_size *= input_gradient_X->shape[i];
    }
    size_t blocks = (input_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (mode == 0) {
        if (stream_handle)
            pad_constant_gradient_kernel<<<blocks, THREADS_PER_BLOCK, 0,
                                           *(cudaStream_t *)
                                                stream_handle->handle>>>(
                (const float *)(output_gradient_Y->data),
                (float *)(input_gradient_X->data), N, C, H, W, begin_p[0],
                begin_p[1], begin_p[2], begin_p[3], out_N, out_C, out_H, out_W);
        else
            pad_constant_gradient_kernel<<<blocks, THREADS_PER_BLOCK>>>(
                (const float *)(output_gradient_Y->data),
                (float *)(input_gradient_X->data), N, C, H, W, begin_p[0],
                begin_p[1], begin_p[2], begin_p[3], out_N, out_C, out_H, out_W);
    }
    return 0;
}