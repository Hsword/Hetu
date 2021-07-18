#include <cassert>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
#include <type_traits>
#include <sys/time.h>

#include "dnnl.hpp"

#include "../common/c_runtime_api.h"
#include "dnnl_runtime.h"

using namespace dnnl;
using namespace std;

extern "C" int DnnlMatrixElementwiseAdd(const DLArrayHandle matA,
                                        const DLArrayHandle matB,
                                        DLArrayHandle output) {
    // use openmp to fit different shape of mat_a,mat_b.such as [3,5]+[5,]=[3,5]
    size_t A_size = 1;
    for (index_t i = 0; i < matA->ndim; i++) {
        A_size *= matA->shape[i];
    }
    size_t B_size = 1;
    for (index_t i = 0; i < matB->ndim; i++) {
        B_size *= matB->shape[i];
    }
    size_t output_size = A_size > B_size ? A_size : B_size;
    int *A_strides = new int[matA->ndim];
    int *B_strides = new int[matB->ndim];
    size_t tmp_size = 1;
    for (int i = matA->ndim - 1; i >= 0; --i) {
        A_strides[i] = tmp_size;
        tmp_size *= matA->shape[i];
    }
    tmp_size = 1;
    for (int i = matB->ndim - 1; i >= 0; --i) {
        B_strides[i] = tmp_size;
        tmp_size *= matB->shape[i];
    }
    const float *A_data = (const float *)matA->data;
    const float *B_data = (const float *)matB->data;
    float *out_data = (float *)output->data;

    if (A_size == B_size) {
        dnnl_stream_init();
        vector<long int> shape, format, one;
        for (int i = 0; i < matA->ndim; i++)
            shape.push_back(matA->shape[i]);
        format.resize(matA->ndim);
        format[(matA->ndim) - 1] = 1;
        for (int i = format.size() - 2; i >= 0; i--)
            format[i] = format[i + 1] * shape[i + 1];
        auto mat_md = memory::desc(shape, memory::data_type::f32, format);
        auto srcA_mem = memory(mat_md, eng, matA->data);
        auto srcB_mem = memory(mat_md, eng, matB->data);
        auto dst_mem = memory(mat_md, eng, output->data);
        auto AddElewise_d =
            binary::desc(algorithm::binary_add, mat_md, mat_md, mat_md);
        auto AddElewise_pd = binary::primitive_desc(AddElewise_d, eng);
        auto AddElewise = binary(AddElewise_pd);

        AddElewise.execute(engine_stream, {{DNNL_ARG_SRC_0, srcA_mem},
                                           {DNNL_ARG_SRC_1, srcB_mem},
                                           {DNNL_ARG_DST, dst_mem}});
        engine_stream.wait();

    }

    else if (A_size > B_size) {
#pragma omp parallel for
        for (size_t i = 0; i < output_size; i++) {
            size_t s_ind = 0;
            uint temp = i;
            for (int j = 0; j < matA->ndim; ++j) {
                uint adder = temp / A_strides[j];
                if (matA->ndim - j <= matB->ndim
                    && matB->shape[j - (matA->ndim - matB->ndim)] > 1) {
                    s_ind += B_strides[j - (matA->ndim - matB->ndim)] * adder;
                }
                temp %= A_strides[j];
            }
            out_data[i] = A_data[i] + B_data[s_ind];
        }
    } else {
#pragma omp parallel for
        for (size_t i = 0; i < output_size; i++) {
            size_t s_ind = 0;
            uint temp = i;
            for (int j = 0; j < matB->ndim; ++j) {
                uint adder = temp / B_strides[j];
                if (matB->ndim - j <= matA->ndim
                    && matA->shape[j - (matB->ndim - matA->ndim)] > 1) {
                    s_ind += A_strides[j - (matB->ndim - matA->ndim)] * adder;
                }
                temp %= B_strides[j];
            }
            out_data[i] = B_data[i] + A_data[s_ind];
        }
    }
    delete A_strides;
    delete B_strides;
    return 0;
}
