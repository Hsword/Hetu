#include <cctype>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
#include <type_traits>
#include <omp.h>
#include "dnnl.hpp"

#include "../common/c_runtime_api.h"
#include "dnnl_runtime.h"

int cpu_Pad(const DLArrayHandle input_X, DLArrayHandle output_Y, int *paddings,
            int pad_len, size_t mode = 0, float constant_values = 0) {
    assert(input_X->ndim == 4 && output_Y->ndim == 4);

    int len = pad_len;
    int endpoint[8];
    for (int i = 0; i < 4; i++) {
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
    // int in_N = input_X->shape[0];
    int in_C = input_X->shape[1];
    int in_H = input_X->shape[2];
    int in_W = input_X->shape[3];
    // int N = output_Y->shape[0];
    int C = output_Y->shape[1];
    int H = output_Y->shape[2];
    int W = output_Y->shape[3];
    int output_size = 1;
    for (int i = 0; i < output_Y->ndim; i++)
        output_size *= output_Y->shape[i];
    float *input = (float *)(input_X->data);
    float *output = (float *)(output_Y->data);

    if (mode == 0) {
#pragma omp parallel for
        for (int i = 0; i < output_size; i++)
            output[i] = constant_values;

        for (int i = endpoint[0]; i < endpoint[1]; i++)
            for (int j = endpoint[2]; j < endpoint[3]; j++)
                for (int k = endpoint[4]; k < endpoint[5]; k++)
                    for (int l = endpoint[6]; l < endpoint[7]; l++) {
                        output[i * (C * H * W) + j * (H * W) + k * W + l] =
                            input[(i - endpoint[0]) * (in_C * in_H * in_W)
                                  + (j - endpoint[2]) * (in_H * in_W)
                                  + (k - endpoint[4]) * in_W
                                  + (l - endpoint[6])];
                    }
    }
    return 0;
}

int cpu_Pad_Gradient(const DLArrayHandle output_gradient_Y,
                     DLArrayHandle input_gradient_X, int *paddings, int pad_len,
                     size_t mode) {
    assert(input_gradient_X->ndim == 4 && output_gradient_Y->ndim == 4);
    int len = pad_len;
    int begin_p[4];
    int N = input_gradient_X->shape[0];
    int C = input_gradient_X->shape[1];
    int H = input_gradient_X->shape[2];
    int W = input_gradient_X->shape[3];

    // int out_N = output_gradient_Y->shape[0];
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
    float *input = (float *)input_gradient_X->data;
    float *output = (float *)output_gradient_Y->data;
#pragma omp parallel for
    for (int i = begin_p[0]; i < N + begin_p[0]; i++)
        for (int j = begin_p[1]; j - begin_p[1] < C; j++)
            for (int k = begin_p[2]; k - begin_p[2] < H; k++)
                for (int l = begin_p[3]; l - begin_p[3] < W; l++)
                    input[(i - begin_p[0]) * (C * H * W)
                          + (j - begin_p[1]) * (H * W) + (k - begin_p[2]) * W
                          + (l - begin_p[3])] =
                        output[i * (out_C * out_H * out_W) + j * (out_H * out_W)
                               + k * out_W + l];
    return 0;
}
