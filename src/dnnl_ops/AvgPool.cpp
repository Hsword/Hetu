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

extern "C" int DnnlAvgPool(const DLArrayHandle input, const int kernel_H,
                           const int kernel_W, DLArrayHandle output,
                           const int padding, const int stride) {
    //	engine eng(engine::kind::cpu, 0);
    //	stream engine_stream(eng);
    dnnl_stream_init();

    int input_N = input->shape[0];
    int input_C = input->shape[1];
    int input_H = input->shape[2];
    int input_W = input->shape[3];
    int pooled_H = output->shape[2];
    int pooled_W = output->shape[3];

    memory::dims pool_src_tz = {input_N, input_C, input_H, input_W};
    memory::dims pool_kernel = {kernel_H, kernel_W};
    memory::dims pool_strides = {stride, stride};
    memory::dims pool_padding = {padding, padding};
    auto pool_src_md = memory::desc({pool_src_tz}, memory::data_type::f32,
                                    memory::format_tag::nchw);
    // dst memory
    memory::dims pool_dst_tz = {input_N, input_C, pooled_H, pooled_W};

    auto pool_dst_md = memory::desc({pool_dst_tz}, memory::data_type::f32,
                                    memory::format_tag::nchw);

    auto input_data = memory(pool_src_md, eng, (void *)input->data);
    auto output_data = memory(pool_dst_md, eng, (void *)output->data);

    //[Create pooling primitive]
    auto pool_desc = pooling_forward::desc(
        prop_kind::forward_inference, algorithm::pooling_avg_include_padding,
        pool_src_md, pool_dst_md, pool_strides, pool_kernel, pool_padding,
        pool_padding);

    auto pool_pd = pooling_forward::primitive_desc(pool_desc, eng);
    auto pool_p = pooling_forward(pool_pd);

    //	stream s(eng);
    pool_p.execute(engine_stream,
                   {{DNNL_ARG_SRC, input_data}, {DNNL_ARG_DST, output_data}});
    engine_stream.wait();
    return 0;
}

extern "C" int DnnlAvgPool_Gradient(const DLArrayHandle gradient_Y,
                                    const int kernel_H, const int kernel_W,
                                    DLArrayHandle gradient_X, const int padding,
                                    const int stride) {
    dnnl_stream_init();
    auto N = gradient_Y->shape[0];
    auto C = gradient_Y->shape[1];
    auto H = gradient_Y->shape[2];
    auto W = gradient_Y->shape[3];

    auto pooled_H = gradient_X->shape[2];
    auto pooled_W = gradient_X->shape[3];

    auto gradient_Y_md = memory::desc({N, C, H, W}, memory::data_type::f32,
                                      memory::format_tag::nchw);
    auto gradient_X_md =
        memory::desc({N, C, pooled_H, pooled_W}, memory::data_type::f32,
                     memory::format_tag::nchw);

    auto gradient_Y_mem = memory(gradient_Y_md, eng, (void *)gradient_Y->data);
    auto gradient_X_mem = memory(gradient_X_md, eng, (void *)gradient_X->data);

    auto pooling_backward_d = pooling_backward::desc(
        algorithm::pooling_avg_include_padding, gradient_X_md, gradient_Y_md,
        {stride, stride}, {kernel_H, kernel_W}, {padding, padding},
        {padding, padding});

    // forward
    auto pool_desc = pooling_forward::desc(
        prop_kind::forward_training, algorithm::pooling_avg_include_padding,
        gradient_X_md, gradient_Y_md, {stride, stride}, {kernel_H, kernel_W},
        {padding, padding}, {padding, padding});

    auto pool_pd = pooling_forward::primitive_desc(pool_desc, eng);
    // forward

    auto pooling_backward_pd =
        pooling_backward::primitive_desc(pooling_backward_d, eng, pool_pd);
    auto pooling_backward_p = pooling_backward(pooling_backward_pd);

    //	stream s(eng);
    pooling_backward_p.execute(engine_stream,
                               {{DNNL_ARG_DIFF_SRC, gradient_X_mem},
                                {DNNL_ARG_DIFF_DST, gradient_Y_mem}});
    return 0;
}