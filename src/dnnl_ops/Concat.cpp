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

extern "C" int DnnlConcat(const DLArrayHandle input_x,
                          const DLArrayHandle input_y, DLArrayHandle output,
                          int axis = 0) {
    dnnl_stream_init();
    assert(input_x->ndim == input_y->ndim);
    assert(input_y->ndim == output->ndim);

    int now_ndim = input_x->ndim;
    for (int i = 0; i < now_ndim; i++) {
        if (i != axis) {
            assert(input_x->shape[i] == input_y->shape[i]);
            assert(input_y->shape[i] == output->shape[i]);
        } else {
            assert(input_x->shape[i] + input_y->shape[i] == output->shape[i]);
        }
    }

    std::vector<long int> shape1, format1, shape2, format2, shape3, format3;
    for (int i = 0; i < input_x->ndim; i++) {
        shape1.push_back(input_x->shape[i]);
        shape2.push_back(input_y->shape[i]);
        shape3.push_back(output->shape[i]);
    }
    format1.resize(input_x->ndim);
    format2.resize(input_y->ndim);
    format3.resize(output->ndim);
    format1[(input_x->ndim) - 1] = format2[(input_y->ndim) - 1] =
        format3[(output->ndim) - 1] = 1;
    for (int i = format1.size() - 2; i >= 0; i--) {
        format1[i] = format1[i + 1] * shape1[i + 1];
        format2[i] = format2[i + 1] * shape2[i + 1];
        format3[i] = format3[i + 1] * shape3[i + 1];
    }
    auto matA_md = memory::desc(shape1, memory::data_type::f32, format1);
    auto matB_md = memory::desc(shape2, memory::data_type::f32, format2);
    auto src_memA = memory(matA_md, eng, input_x->data);
    auto src_memB = memory(matB_md, eng, input_y->data);
    auto dst_md = memory::desc(shape3, memory::data_type::f32, format3);
    auto dst_mem = memory(dst_md, eng, output->data);

    std::vector<memory::desc> srcs_md;
    std::vector<memory> srcs_mem;
    srcs_md.push_back(matA_md);
    srcs_md.push_back(matB_md);
    srcs_mem.push_back(src_memA);
    srcs_mem.push_back(src_memB);

    auto concat_pd = concat::primitive_desc(dst_md, axis, srcs_md, eng);
    auto concat_e = concat(concat_pd);
    concat_e.execute(engine_stream, {{DNNL_ARG_DST, dst_mem},
                                     {DNNL_ARG_MULTIPLE_SRC + 0, srcs_mem[0]},
                                     {DNNL_ARG_MULTIPLE_SRC + 1, srcs_mem[1]}});
    engine_stream.wait();
    return 0;
}

extern "C" int cpu_Concat_Gradient(const DLArrayHandle output_gradient,
                                   DLArrayHandle input_gradient, int axis = 0,
                                   int id = 0) {
    assert(output_gradient->ndim == input_gradient->ndim);
    for (int i = 0; i < input_gradient->ndim; i++)
        if (i != axis)
            assert(input_gradient->shape[i] == output_gradient->shape[i]);

    int start_position;
    int end_position;
    if (id == 0) {
        start_position = 0;
        end_position = 1;
        for (int i = 0; i < output_gradient->ndim; i++)
            if (i != axis)
                end_position *= output_gradient->shape[i];
            else
                end_position *= input_gradient->shape[i];
    } else {
        start_position = 1;
        end_position = 1;
        for (int i = 0; i < output_gradient->ndim; i++) {
            end_position *= output_gradient->shape[i];
            if (i != axis)
                start_position *= output_gradient->shape[i];
            else
                start_position *=
                    (output_gradient->shape[i]) - (input_gradient->shape[i]);
        }
    }

    float *output = (float *)(output_gradient->data);
    float *input = (float *)(input_gradient->data);
#pragma omp parallel for
    for (int i = start_position; i < end_position; i++)
        input[i] = output[i];

    return 0;
}
