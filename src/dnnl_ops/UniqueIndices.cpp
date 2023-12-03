
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
#include <type_traits>
#include <sys/time.h>
#include <map>

#include "dnnl.hpp"

#include "../common/c_runtime_api.h"
#include "dnnl_runtime.h"
using namespace dnnl;
using namespace std;

int cpu_UniqueIndices(const DLArrayHandle indices, DLArrayHandle output,
                      DLArrayHandle idoffsets) {
    map<int, vector<size_t>> idx2map;
    size_t dup_index_size = 1;
    for (int i = 0; i < indices->ndim; ++i) {
        dup_index_size *= indices->shape[i];
    }
    const int *dup_index = (const int *)indices->data;
    int *out_ind = (int *)output->data;
    int *punique_size = (int *)(idoffsets->data);
    int *id_offset = punique_size + 1;
    int *id_length = id_offset + dup_index_size;
    for (size_t i = 0; i < dup_index_size; ++i) {
        idx2map[dup_index[i]].emplace_back(i);
        out_ind[i] = -1;
    }

    size_t index_size = idx2map.size();
    *punique_size = (int)index_size;

    vector<pair<int, vector<size_t>>> idx2vector(idx2map.begin(),
                                                 idx2map.end());

    size_t cnt = 0;
    id_length[0] = 0;
    for (size_t i = 0; i < index_size; ++i) {
        auto cur_pair = idx2vector[i];
        out_ind[i] = cur_pair.first;
        auto &cur_vector = cur_pair.second;
        size_t cur_size = cur_vector.size();
        for (size_t j = 0; j < cur_size; ++j) {
            id_offset[cnt + j] = cur_vector[j];
        }
        cnt += cur_size;
        id_length[i + 1] = cnt;
    }

    return 0;
}

int cpu_DedupLookup(const DLArrayHandle lookups, const DLArrayHandle idoffsets,
                    DLArrayHandle output) {
    int ind_size = 1;
    for (int i = 0; i < lookups->ndim - 1; ++i) {
        ind_size *= lookups->shape[i];
    }
    size_t width = lookups->shape[lookups->ndim - 1];
    const int *punique_size = (const int *)idoffsets->data;
    const int *id_offset = punique_size + 1;
    const int *id_length = id_offset + ind_size;
    const float *in_val_data = (const float *)lookups->data;
    float *out_val_data = (float *)output->data;
    const int unique_size = *punique_size;
    size_t rowsize = sizeof(float) * width;

#pragma omp parallel for
    for (int i = 0; i < ind_size; ++i) {
        float *cur_output = out_val_data + i * width;
        if (i < unique_size) {
            const float *cur_input =
                in_val_data + id_offset[id_length[i]] * width;
            memcpy(cur_output, cur_input, rowsize);
        } else {
            memset(cur_output, 0, rowsize);
        }
    }
    return 0;
}

int cpu_DedupGrad(const DLArrayHandle grad, const DLArrayHandle idoffsets,
                  DLArrayHandle output) {
    int ind_size = 1;
    for (int i = 0; i < grad->ndim - 1; ++i) {
        ind_size *= grad->shape[i];
    }
    int width = grad->shape[grad->ndim - 1];
    const int *punique_size = (const int *)idoffsets->data;
    const int *id_offset = punique_size + 1;
    const int *id_length = id_offset + ind_size;
    const float *in_val_data = (const float *)grad->data;
    float *out_val_data = (float *)output->data;
    const int unique_size = *punique_size;
    size_t rowsize = sizeof(float) * width;

#pragma omp parallel for
    for (int i = 0; i < ind_size; ++i) {
        float *cur_output = out_val_data + i * width;
        memset(cur_output, 0, rowsize);
        if (i < unique_size) {
            int l = id_length[i], r = id_length[i + 1];
            for (int j = l; j < r; ++j) {
                const float *cur_input = in_val_data + id_offset[j] * width;
                for (int k = 0; k < width; ++k) {
                    cur_output[k] += cur_input[k];
                }
            }
        }
    }
    return 0;
}
