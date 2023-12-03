#include <cctype>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <type_traits>
#include <omp.h>
#include "dnnl.hpp"

#include "../common/c_runtime_api.h"
#include "../common/random.h"
#include "dnnl_runtime.h"

extern "C" int cpu_Dropout(const DLArrayHandle input_X, const float dropout,
                           DLArrayHandle output_Y) {
    HetuRandomState &heturs = GetRandomState(1);
    unsigned long long sequence[2] = {heturs.seed, heturs.seqnum};
    std::seed_seq cur_seed_seq(sequence, sequence + 2);
    std::default_random_engine generator(cur_seed_seq);
    std::uniform_real_distribution<float> uniform_dist(0.0, 1.0);

    float *output = (float *)output_Y->data;
    float *input = (float *)input_X->data;
    int data_size = 1;
    assert(input_X->ndim == output_Y->ndim);
    for (int i = 0; i < input_X->ndim; i++) {
        assert(input_X->shape[i] == output_Y->shape[i]);
        data_size *= input_X->shape[i];
    }
    for (int i = 0; i < data_size; i++) {
        if (uniform_dist(generator) > dropout)
            output[i] = 0;
        else
            output[i] = input[i] * (1 / dropout);
    }
    return 0;
}

extern "C" int cpu_Dropout_Gradient(const DLArrayHandle output_Y,
                                    const float dropout, DLArrayHandle input_X,
                                    unsigned long long seqnum) {
    unsigned long long sequence[2] = {GetSeed(), seqnum};
    std::seed_seq cur_seed_seq(sequence, sequence + 2);
    std::default_random_engine generator(cur_seed_seq);
    std::uniform_real_distribution<float> uniform_dist(0.0, 1.0);

    float *output = (float *)output_Y->data;
    float *input = (float *)input_X->data;
    int data_size = 1;
    assert(input_X->ndim == output_Y->ndim);
    for (int i = 0; i < input_X->ndim; i++) {
        assert(input_X->shape[i] == output_Y->shape[i]);
        data_size *= input_X->shape[i];
    }
    for (int i = 0; i < data_size; i++) {
        if (uniform_dist(generator) > dropout)
            input[i] = 0;
        else
            input[i] = output[i] * (1 / dropout);
    }
    return 0;
}
