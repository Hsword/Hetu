#include <cctype>
#include <cstring>
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

int cpu_NormalInit(DLArrayHandle arr, const float mean, const float stddev,
                   unsigned long long seed) {
    size_t size = 1;
    for (index_t i = 0; i < arr->ndim; i++) {
        size *= arr->shape[i];
    }
    float *arr_data = (float *)arr->data;

    std::normal_distribution<float> normal_dist(mean, stddev);
    std::default_random_engine generator(seed);
    for (size_t j = 0; j < size; j++) {
        arr_data[j] = normal_dist(generator);
    }
    return 0;
}

int cpu_UniformInit(DLArrayHandle arr, const float lb, const float ub,
                    unsigned long long seed) {
    size_t size = 1;
    for (index_t i = 0; i < arr->ndim; i++) {
        size *= arr->shape[i];
    }
    float *arr_data = (float *)arr->data;

    std::uniform_real_distribution<float> uniform_dist(lb, ub);
    std::default_random_engine generator(seed);
    for (size_t j = 0; j < size; j++) {
        arr_data[j] = uniform_dist(generator);
    }
    return 0;
}

int cpu_TruncatedNormalInit(DLArrayHandle arr, const float mean,
                            const float stddev, unsigned long long seed) {
    size_t size = 1;
    for (index_t i = 0; i < arr->ndim; i++) {
        size *= arr->shape[i];
    }
    float *arr_data = (float *)arr->data;

    std::normal_distribution<float> truncated_normal_dist(mean, stddev);
    float upper_limit = mean + 2 * stddev;
    float lower_limit = mean - 2 * stddev;
    std::default_random_engine generator(seed);
    for (size_t j = 0; j < size; j++) {
        float temp = truncated_normal_dist(generator);
        while (temp > upper_limit || temp < lower_limit)
            temp = truncated_normal_dist(generator);
        arr_data[j] = temp;
    }
    return 0;
}