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

extern "C" int cpu_EmbeddingLookup(const DLArrayHandle in_mat,
                                   const DLArrayHandle ids,
                                   DLArrayHandle out_mat) {
    const float *embed = (const float *)(in_mat->data);
    const int *index = (const int *)(ids->data);
    float *output = (float *)(out_mat->data);
    assert(in_mat->ndim == 2);
    int nrow = in_mat->shape[0];
    size_t width = in_mat->shape[1];
    size_t entry_size = width * sizeof(float);
    size_t idx_size = 1;
    for (int i = 0; i < ids->ndim; ++i)
        idx_size *= ids->shape[i];

#pragma omp parallel for
    for (size_t i = 0; i < idx_size; ++i) {
        int id = index[i];
        if (id < 0 || id > nrow)
            continue;
        memcpy(output + i * width, embed + index[i] * width, entry_size);
    }
    return 0;
}