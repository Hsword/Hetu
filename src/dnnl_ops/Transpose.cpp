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

extern "C" int cpu_Transpose(const DLArrayHandle in_arr, DLArrayHandle out_arr,
                             int *perm) {
    const float *input = (const float *)(in_arr->data);
    float *output = (float *)(out_arr->data);
    int64_t *in_dims = in_arr->shape;
    int64_t *out_dims = out_arr->shape;
    uint ndim = uint(in_arr->ndim);

    uint *in_strides = (uint *)malloc(ndim * sizeof(uint));
    uint *out_strides = (uint *)malloc(ndim * sizeof(uint));
    uint in_stride = 1;
    uint out_stride = 1;
    for (int i = ndim - 1; i >= 0; --i) {
        in_strides[i] = in_stride;
        out_strides[i] = out_stride;
        in_stride *= uint(in_dims[i]);
        out_stride *= uint(out_dims[i]);
    }

#pragma omp parallel for
    for (uint o_idx = 0; o_idx < out_stride; ++o_idx) {
        uint i_idx = 0;
        uint temp = o_idx;
        for (uint i = 0; i < ndim; ++i) {
            const uint ratio = temp / out_strides[i];
            temp -= ratio * out_strides[i];
            i_idx += ratio * in_strides[perm[i]];
        }
        output[o_idx] = input[i_idx];
    }
    free(in_strides);
    free(out_strides);
    return 0;
}