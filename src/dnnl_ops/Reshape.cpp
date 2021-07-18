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

extern "C" int cpu_Reshape(const DLArrayHandle in_arr, DLArrayHandle out_arr) {
    int input_size = 1;
    int output_size = 1;
    float *input = (float *)(in_arr->data);
    float *output = (float *)(out_arr->data);
    for (int i = 0; i < in_arr->ndim; i++)
        input_size *= in_arr->shape[i];
    for (int i = 0; i < out_arr->ndim; i++)
        output_size *= out_arr->shape[i];

    assert(input_size == output_size);
#pragma omp parallel for
    for (int i = 0; i < input_size; i++)
        output[i] = input[i];
    return 0;
}