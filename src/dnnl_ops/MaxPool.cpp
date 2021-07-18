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

extern "C" int DnnlMaxPool(const DLArrayHandle input, const int kernel_H,
                           const int kernel_W, DLArrayHandle output,
                           const int padding, const int stride) {
    dnnl_stream_init();
    using tag = memory::format_tag;
    using dt = memory::data_type;

    int input_N = input->shape[0];
    int input_C = input->shape[1];
    int input_H = input->shape[2];
    int input_W = input->shape[3];
    int pooled_H = output->shape[2];
    int pooled_W = output->shape[3];

    // src memory
    memory::dims pool_src_tz = {input_N, input_C, input_H, input_W};
    memory::dims pool_kernel = {kernel_H, kernel_W};
    memory::dims pool_strides = {stride, stride};
    memory::dims pool_padding = {padding, padding};
    auto pool_src_md = memory::desc({pool_src_tz}, dt::f32, tag::nchw);
    // dst memory
    memory::dims pool_dst_tz = {input_N, input_C, pooled_H, pooled_W};

    auto pool_dst_md = memory::desc({pool_dst_tz}, dt::f32, tag::nchw);

    auto input_data = memory(pool_src_md, eng, (void *)input->data);
    auto output_data = memory(pool_dst_md, eng, (void *)output->data);

    //[Create pooling primitive]
    auto pool_desc = pooling_forward::desc(
        prop_kind::forward_inference, algorithm::pooling_max, pool_src_md,
        pool_dst_md, pool_strides, pool_kernel, pool_padding, pool_padding);

    auto pool_pd = pooling_forward::primitive_desc(pool_desc, eng);
    auto pool_p = pooling_forward(pool_pd);

    pool_p.execute(engine_stream,
                   {{DNNL_ARG_SRC, input_data}, {DNNL_ARG_DST, output_data}});

    return 0;
}

extern "C" int DnnlMaxPool_Gradient(const DLArrayHandle input,
                                    const DLArrayHandle input_grad,
                                    const int kernel_H, const int kernel_W,
                                    DLArrayHandle output_grad,
                                    const int padding, const int stride) {
    int input_N = input->shape[0];
    int input_C = input->shape[1];
    int input_H = input->shape[2];
    int input_W = input->shape[3];
    int pooled_H = input_grad->shape[2];
    int pooled_W = input_grad->shape[3];

    dnnl_stream_init();

    auto input_md =
        memory::desc({input_N, input_C, input_H, input_W},
                     memory::data_type::f32, memory::format_tag::nchw);
    auto input_grad_md =
        memory::desc({input_N, input_C, pooled_H, pooled_W},
                     memory::data_type::f32, memory::format_tag::nchw);
    auto output_grad_md =
        memory::desc({input_N, input_C, input_H, input_W},
                     memory::data_type::f32, memory::format_tag::nchw);

    auto input_mem = memory(input_md, eng, (void *)input->data);
    auto input_grad_mem = memory(input_grad_md, eng, (void *)input_grad->data);
    auto output_grad_mem =
        memory(output_grad_md, eng, (void *)output_grad->data);

    auto pooling_backward_d = pooling_backward::desc(
        algorithm::pooling_max, output_grad_md, input_grad_md, {stride, stride},
        {kernel_H, kernel_W}, {padding, padding}, {padding, padding});

    // forward
    auto input_forward_md =
        memory::desc({input_N, input_C, pooled_H, pooled_W},
                     memory::data_type::f32, memory::format_tag::nchw);
    auto input_forward_mem = memory(input_forward_md, eng);
    auto pool_desc = pooling_forward::desc(
        prop_kind::forward_training, algorithm::pooling_max, input_md,
        input_forward_md, {stride, stride}, {kernel_H, kernel_W},
        {padding, padding}, {padding, padding});
    auto pool_pd = pooling_forward::primitive_desc(pool_desc, eng);
    auto workspace_mem = memory(pool_pd.workspace_desc(), eng);
    auto pool_p = pooling_forward(pool_pd);
    pool_p.execute(engine_stream, {{DNNL_ARG_SRC, input_mem},
                                   {DNNL_ARG_DST, input_forward_mem},
                                   {DNNL_ARG_WORKSPACE, workspace_mem}});

    // backward
    auto pooling_backward_pd =
        pooling_backward::primitive_desc(pooling_backward_d, eng, pool_pd);

    auto pooling_backward_p = pooling_backward(pooling_backward_pd);

    pooling_backward_p.execute(engine_stream,
                               {{DNNL_ARG_DIFF_SRC, output_grad_mem},
                                {DNNL_ARG_WORKSPACE, workspace_mem},
                                {DNNL_ARG_DIFF_DST, input_grad_mem}});

    engine_stream.wait();
    return 1;
}
