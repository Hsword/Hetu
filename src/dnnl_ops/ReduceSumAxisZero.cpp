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

extern "C" int cpu_ReduceSumAxisZero(const DLArrayHandle input,
                                     DLArrayHandle output) {
    for (int i = 1; i < (input->ndim); i++) {
        assert((input->shape[i]) == (output->shape[i - 1]));
    }
    const float *input_data = (const float *)input->data;
    float *output_data = (float *)output->data;
    size_t output_size = 1;
    for (index_t i = 0; i < (output->ndim); i++) {
        output_size *= (output->shape[i]);
    }
    size_t N = input->shape[0];
#pragma omp parallel for
    for (size_t j = 0; j < output_size; j++)
        output_data[j] = 0;

#pragma omp parallel for
    for (size_t i = 0; i < N * output_size; i++)
        output_data[i % output_size] += input_data[i];
    return 0;
}