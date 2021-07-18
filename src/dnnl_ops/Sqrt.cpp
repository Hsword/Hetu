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

extern "C" int DnnlSqrt(const DLArrayHandle input, DLArrayHandle output) {
    // engine eng(engine::kind::cpu, 0);
    // stream engine_stream(eng);
    dnnl_stream_init();

    vector<long int> shape, format;
    for (int i = 0; i < input->ndim; i++)
        shape.push_back(input->shape[i]);
    format.resize(input->ndim);
    format[(input->ndim) - 1] = 1;
    for (int i = format.size() - 2; i >= 0; i--)
        format[i] = format[i + 1] * shape[i + 1];
    auto mat_md = memory::desc(shape, memory::data_type::f32, format);

    auto src_mem = memory(mat_md, eng, input->data);
    auto dst_mem = memory(mat_md, eng, output->data);

    auto Sqrt_d = eltwise_forward::desc(
        prop_kind::forward_training, algorithm::eltwise_sqrt, mat_md, 0.f, 0.f);
    auto Sqrt_pd = eltwise_forward::primitive_desc(Sqrt_d, eng);
    auto Sqrt = eltwise_forward(Sqrt_pd);

    Sqrt.execute(engine_stream,
                 {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_DST, dst_mem}});
    engine_stream.wait();
    return 0;
}

extern "C" int DnnlReciprocalSqrt(const DLArrayHandle input,
                                  DLArrayHandle output) {
    // engine eng(engine::kind::cpu, 0);
    // stream engine_stream(eng);
    dnnl_stream_init();

    size_t size = 1;
    vector<long int> shape, format;
    for (int i = 0; i < input->ndim; i++) {
        shape.push_back(input->shape[i]);
        size *= input->shape[i];
    }
    format.resize(input->ndim);
    format[(input->ndim) - 1] = 1;
    for (int i = format.size() - 2; i >= 0; i--)
        format[i] = format[i + 1] * shape[i + 1];

    float *temp;
    temp = new float[size];

    auto mat_md = memory::desc(shape, memory::data_type::f32, format);
    auto src_mem = memory(mat_md, eng, input->data);
    auto temp_mem = memory(mat_md, eng, temp);
    auto dst_mem = memory(mat_md, eng, output->data);

    auto Sqrt_d = eltwise_forward::desc(prop_kind::forward_training,
                                        algorithm::eltwise_sqrt, mat_md);
    auto Sqrt_pd = eltwise_forward::primitive_desc(Sqrt_d, eng);
    auto Sqrt = eltwise_forward(Sqrt_pd);

    Sqrt.execute(engine_stream,
                 {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_DST, temp_mem}});
    engine_stream.wait();

    auto DivideConst_d = eltwise_forward::desc(
        prop_kind::forward_training, algorithm::eltwise_pow, mat_md, 1, -1.f);
    auto DivideConst_pd = eltwise_forward::primitive_desc(DivideConst_d, eng);
    auto DivideConst = eltwise_forward(DivideConst_pd);

    DivideConst.execute(engine_stream,
                        {{DNNL_ARG_SRC, temp_mem}, {DNNL_ARG_DST, dst_mem}});
    engine_stream.wait();

    delete temp;
    return 0;
}