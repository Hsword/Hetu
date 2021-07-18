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

extern "C" int DnnlRelu(const DLArrayHandle input, DLArrayHandle output) {
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

    auto Relu_d = eltwise_forward::desc(
        prop_kind::forward_training, algorithm::eltwise_relu, mat_md, 0.f, 0.f);
    auto Relu_pd = eltwise_forward::primitive_desc(Relu_d, eng);
    auto Relu = eltwise_forward(Relu_pd);

    Relu.execute(engine_stream,
                 {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_DST, dst_mem}});
    engine_stream.wait();
    return 0;
}

extern "C" int DnnlRelu_Gradient(const DLArrayHandle input,
                                 const DLArrayHandle in_grad,
                                 DLArrayHandle output) {
    dnnl_stream_init();

    vector<long int> shape, format;
    for (int i = 0; i < input->ndim; i++)
        shape.push_back(input->shape[i]);
    format.resize(input->ndim);
    format[(input->ndim) - 1] = 1;
    for (int i = format.size() - 2; i >= 0; i--)
        format[i] = format[i + 1] * shape[i + 1];
    auto input_md = memory::desc(shape, memory::data_type::f32, format);
    auto in_grad_md = memory::desc(shape, memory::data_type::f32, format);
    auto output_md = memory::desc(shape, memory::data_type::f32, format);

    auto input_mem = memory(input_md, eng, input->data);
    auto in_grad_mem = memory(in_grad_md, eng, in_grad->data);
    auto output_mem = memory(output_md, eng, output->data);

    // forward
    auto Relu_d = eltwise_forward::desc(prop_kind::forward_training,
                                        algorithm::eltwise_relu, input_md);
    auto Relu_pd = eltwise_forward::primitive_desc(Relu_d, eng);

    // backward
    auto Relu_gradient_d =
        eltwise_backward::desc(algorithm::eltwise_relu, in_grad_md, output_md);
    auto Relu_gradient_pd =
        eltwise_backward::primitive_desc(Relu_gradient_d, eng, Relu_pd);
    auto Relu_gradient_p = eltwise_backward(Relu_gradient_pd);
    Relu_gradient_p.execute(engine_stream, {{DNNL_ARG_SRC, input_mem},
                                            {DNNL_ARG_DIFF_DST, in_grad_mem},
                                            {DNNL_ARG_DIFF_SRC, output_mem}});
    engine_stream.wait();
    return 0;
}
