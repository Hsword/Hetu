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

extern "C" int cpu_Dropout(const DLArrayHandle input_X, const float dropout,
                           DLArrayHandle output_Y) {
    int seed = 233;
    srand(seed);

    float *output = (float *)output_Y->data;
    float *input = (float *)input_X->data;
    int data_size = 1;
    assert(input_X->ndim == output_Y->ndim);
    for (int i = 0; i < input_X->ndim; i++) {
        assert(input_X->shape[i] == output_Y->shape[i]);
        data_size *= input_X->shape[i];
    }
    for (int i = 0; i < data_size; i++) {
        if (rand() / (double)RAND_MAX > dropout)
            output[i] = 0;
        else
            output[i] = input[i] * (1 / dropout);
    }
    return 0;
}

extern "C" int cpu_Dropout_Gradient(const DLArrayHandle output_Y,
                                    const float dropout,
                                    DLArrayHandle input_X) {
    int seed = 233;
    srand(seed);

    float *output = (float *)output_Y->data;
    float *input = (float *)input_X->data;
    int data_size = 1;
    assert(input_X->ndim == output_Y->ndim);
    for (int i = 0; i < input_X->ndim; i++) {
        assert(input_X->shape[i] == output_Y->shape[i]);
        data_size *= input_X->shape[i];
    }
    for (int i = 0; i < data_size; i++) {
        if (rand() / (double)RAND_MAX > dropout)
            input[i] = 0;
        else
            input[i] = output[i] * (1 / dropout);
    }
    return 0;
}
