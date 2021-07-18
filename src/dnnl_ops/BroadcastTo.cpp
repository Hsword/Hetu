#include <cassert>
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

using namespace dnnl;

// completed by omp, DNNL does not support this op
extern "C" int cpu_BroadcastTo(const DLArrayHandle in_arr,
                               DLArrayHandle out_arr) {
    for (index_t i = 0; i < in_arr->ndim; i++) {
        assert((in_arr->shape[i]) == (out_arr->shape[i + 1]));
    }
    size_t input_size = 1;
    for (index_t i = 0; i < in_arr->ndim; i++) {
        input_size *= in_arr->shape[i];
    }
    size_t N = (out_arr->shape[0]);
    const float *in_data = (const float *)in_arr->data;
    float *out_data = (float *)out_arr->data;

#pragma omp parallel for
    for (size_t i = 0; i < input_size; i++) {
        float tmp = in_data[i];
#pragma omp parallel for
        for (size_t j = 0; j < N; j++) {
            out_data[j * input_size + i] = tmp;
        }
    }
    return 0;
}