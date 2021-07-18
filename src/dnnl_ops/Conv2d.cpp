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

extern "C" int DnnlConv2d(const DLArrayHandle input_x, DLArrayHandle input_f,
                          DLArrayHandle output, const int padding,
                          const int stride) {
    dnnl_stream_init();

    assert(input_x->ndim == 4);
    assert(input_f->ndim == 4);
    assert(input_x->shape[1] == input_f->shape[1]);

    int N = input_x->shape[0];
    int C = input_x->shape[1];
    int H = input_x->shape[2];
    int W = input_x->shape[3];
    int filter_outChannel = input_f->shape[0];
    int filter_H = input_f->shape[2];
    int filter_W = input_f->shape[3];
    assert((H + 2 * padding - filter_H) % stride == 0);
    assert((W + 2 * padding - filter_W) % stride == 0);
    int out_H = (H + 2 * padding - filter_H) / stride + 1;
    int out_W = (W + 2 * padding - filter_W) / stride + 1;

    auto src_md = memory::desc({N, C, H, W}, memory::data_type::f32,
                               memory::format_tag::nchw);
    auto weights_md =
        memory::desc({filter_outChannel, C, filter_H, filter_W},
                     memory::data_type::f32, memory::format_tag::oihw);
    auto dst_md =
        memory::desc({N, filter_outChannel, out_H, out_W},
                     memory::data_type::f32, memory::format_tag::nchw);

    auto src_mem = memory(src_md, eng, input_x->data);
    auto weights_mem = memory(weights_md, eng, input_f->data);
    auto dst_mem = memory(dst_md, eng, output->data);

    auto conv_src_md = memory::desc({N, C, H, W}, memory::data_type::f32,
                                    memory::format_tag::any);
    auto conv_weights_md =
        memory::desc({filter_outChannel, C, filter_H, filter_W},
                     memory::data_type::f32, memory::format_tag::any);
    auto conv_dst_md =
        memory::desc({N, filter_outChannel, out_H, out_W},
                     memory::data_type::f32, memory::format_tag::any);

    auto conv_src_mem = src_mem;
    auto conv_weights_mem = weights_mem;
    auto conv_dst_mem = dst_mem;

    auto conv_desc = convolution_forward::desc(
        prop_kind::forward, algorithm::convolution_auto, conv_src_md,
        conv_weights_md, conv_dst_md, {stride, stride}, {padding, padding},
        {padding, padding});

    auto conv_prim_desc = convolution_forward::primitive_desc(conv_desc, eng);

    if (conv_prim_desc.src_desc() != src_mem.get_desc()) {
        conv_src_mem = memory(conv_prim_desc.src_desc(), eng);
        reorder(src_mem, conv_src_mem)
            .execute(engine_stream, src_mem, conv_src_mem);
    }
    if (conv_prim_desc.weights_desc() != weights_mem.get_desc()) {
        conv_weights_mem = memory(conv_prim_desc.weights_desc(), eng);
        reorder(weights_mem, conv_weights_mem)
            .execute(engine_stream, weights_mem, conv_weights_mem);
    }
    if (conv_prim_desc.dst_desc() != dst_mem.get_desc()) {
        conv_dst_mem = memory(conv_prim_desc.dst_desc(), eng);
        reorder(dst_mem, conv_dst_mem)
            .execute(engine_stream, dst_mem, conv_dst_mem);
    }

    auto conv = convolution_forward(conv_prim_desc);

    conv.execute(engine_stream, {{DNNL_ARG_SRC, conv_src_mem},
                                 {DNNL_ARG_WEIGHTS, conv_weights_mem},
                                 {DNNL_ARG_DST, conv_dst_mem}});

    if (conv_prim_desc.dst_desc() != dst_mem.get_desc())
        reorder(conv_dst_mem, dst_mem)
            .execute(engine_stream, conv_dst_mem, dst_mem);

    engine_stream.wait();
    return 1;
}

extern "C" int DnnlConv2d_Gradient_of_Data(const DLArrayHandle input_f,
                                           const DLArrayHandle gradient_y,
                                           DLArrayHandle gradient_x,
                                           const int padding,
                                           const int stride) {
    auto input_N = gradient_x->shape[0];
    auto input_C = gradient_x->shape[1];
    auto input_H = gradient_x->shape[2];
    auto input_W = gradient_x->shape[3];
    auto filter_outChannel = input_f->shape[0];
    auto filter_inChannel = input_f->shape[1];
    auto filter_H = input_f->shape[2];
    auto filter_W = input_f->shape[3];
    auto output_N = gradient_y->shape[0];
    auto output_C = gradient_y->shape[1];
    auto output_H = gradient_y->shape[2];
    auto output_W = gradient_y->shape[3];

    dnnl_stream_init();

    auto gradient_x_md =
        memory::desc({input_N, input_C, input_H, input_W},
                     memory::data_type::f32, memory::format_tag::nchw);
    auto input_f_md =
        memory::desc({filter_outChannel, filter_inChannel, filter_H, filter_W},
                     memory::data_type::f32, memory::format_tag::oihw);
    auto gradient_y_md =
        memory::desc({output_N, output_C, output_H, output_W},
                     memory::data_type::f32, memory::format_tag::nchw);

    auto input_f_mem = memory(input_f_md, eng, input_f->data);
    auto gradient_y_mem = memory(gradient_y_md, eng, gradient_y->data);
    auto gradient_x_mem = memory(gradient_x_md, eng, gradient_x->data);

    auto conv_gradient_x_md =
        memory::desc({input_N, input_C, input_H, input_W},
                     memory::data_type::f32, memory::format_tag::any);
    auto conv_input_f_md =
        memory::desc({filter_outChannel, filter_inChannel, filter_H, filter_W},
                     memory::data_type::f32, memory::format_tag::any);
    auto conv_gradient_y_md =
        memory::desc({output_N, output_C, output_H, output_W},
                     memory::data_type::f32, memory::format_tag::any);

    auto conv_input_f_mem = input_f_mem;
    auto conv_gradient_y_mem = gradient_y_mem;
    auto conv_gradient_x_mem = gradient_x_mem;

    auto gradient_data_d = convolution_backward_data::desc(
        algorithm::convolution_auto, conv_gradient_x_md, conv_input_f_md,
        conv_gradient_y_md, {stride, stride}, {padding, padding},
        {padding, padding});
    // forward
    auto conv_desc = convolution_forward::desc(
        prop_kind::forward, algorithm::convolution_auto, conv_gradient_x_md,
        conv_input_f_md, conv_gradient_y_md, {stride, stride},
        {padding, padding}, {padding, padding});
    auto conv_prim_desc = convolution_forward::primitive_desc(conv_desc, eng);
    // forward

    auto gradient_data_pd = convolution_backward_data::primitive_desc(
        gradient_data_d, eng, conv_prim_desc);

    if (gradient_data_pd.weights_desc() != input_f_mem.get_desc()) {
        conv_input_f_mem = memory(gradient_data_pd.weights_desc(), eng);
        reorder(input_f_mem, conv_input_f_mem)
            .execute(engine_stream, input_f_mem, conv_input_f_mem);
    }
    if (gradient_data_pd.diff_dst_desc() != gradient_y_mem.get_desc()) {
        conv_gradient_y_mem = memory(gradient_data_pd.diff_dst_desc(), eng);
        reorder(gradient_y_mem, conv_gradient_y_mem)
            .execute(engine_stream, gradient_y_mem, conv_gradient_y_mem);
    }
    if (gradient_data_pd.diff_src_desc() != gradient_x_mem.get_desc()) {
        conv_gradient_x_mem = memory(gradient_data_pd.diff_src_desc(), eng);
        reorder(gradient_x_mem, conv_gradient_x_mem)
            .execute(engine_stream, gradient_x_mem, conv_gradient_x_mem);
    }

    auto gradient_data = convolution_backward_data(gradient_data_pd);

    gradient_data.execute(engine_stream,
                          {{DNNL_ARG_DIFF_SRC, conv_gradient_x_mem},
                           {DNNL_ARG_WEIGHTS, conv_input_f_mem},
                           {DNNL_ARG_DIFF_DST, conv_gradient_y_mem}});

    if (gradient_data_pd.diff_src_desc() != gradient_x_mem.get_desc())
        reorder(conv_gradient_x_mem, gradient_x_mem)
            .execute(engine_stream, conv_gradient_x_mem, gradient_x_mem);

    engine_stream.wait();
    return 0;
}

extern "C" int DnnlConv2d_Gradient_of_Filter(const DLArrayHandle input_x,
                                             const DLArrayHandle gradient_y,
                                             DLArrayHandle gradient_f,
                                             const int padding,
                                             const int stride) {
    auto input_N = input_x->shape[0];
    auto input_C = input_x->shape[1];
    auto input_H = input_x->shape[2];
    auto input_W = input_x->shape[3];
    auto filter_outChannel = gradient_f->shape[0];
    auto filter_inChannel = gradient_f->shape[1];
    auto filter_H = gradient_f->shape[2];
    auto filter_W = gradient_f->shape[3];
    auto output_N = gradient_y->shape[0];
    auto output_C = gradient_y->shape[1];
    auto output_H = gradient_y->shape[2];
    auto output_W = gradient_y->shape[3];

    dnnl_stream_init();

    auto input_x_md =
        memory::desc({input_N, input_C, input_H, input_W},
                     memory::data_type::f32, memory::format_tag::nchw);
    auto gradient_f_md =
        memory::desc({filter_outChannel, filter_inChannel, filter_H, filter_W},
                     memory::data_type::f32, memory::format_tag::oihw);
    auto gradient_y_md =
        memory::desc({output_N, output_C, output_H, output_W},
                     memory::data_type::f32, memory::format_tag::nchw);

    auto input_x_mem = memory(input_x_md, eng, input_x->data);
    auto gradient_f_mem = memory(gradient_f_md, eng, gradient_f->data);
    auto gradient_y_mem = memory(gradient_y_md, eng, gradient_y->data);

    auto conv_input_x_mem = input_x_mem;
    auto conv_gradient_f_mem = gradient_f_mem;
    auto conv_gradient_y_mem = gradient_y_mem;

    auto conv_input_x_md =
        memory::desc({input_N, input_C, input_H, input_W},
                     memory::data_type::f32, memory::format_tag::any);
    auto conv_gradient_f_md =
        memory::desc({filter_outChannel, filter_inChannel, filter_H, filter_W},
                     memory::data_type::f32, memory::format_tag::any);
    auto conv_gradient_y_md =
        memory::desc({output_N, output_C, output_H, output_W},
                     memory::data_type::f32, memory::format_tag::any);

    auto gradient_filter_d = convolution_backward_weights::desc(
        algorithm::convolution_auto, conv_input_x_md, conv_gradient_f_md,
        conv_gradient_y_md, {stride, stride}, {padding, padding},
        {padding, padding});

    // forward
    auto conv_desc = convolution_forward::desc(
        prop_kind::forward, algorithm::convolution_auto, conv_input_x_md,
        conv_gradient_f_md, conv_gradient_y_md, {stride, stride},
        {padding, padding}, {padding, padding});

    auto conv_prim_desc = convolution_forward::primitive_desc(conv_desc, eng);
    // forward

    auto gradient_filter_pd = convolution_backward_weights::primitive_desc(
        gradient_filter_d, eng, conv_prim_desc);

    if (gradient_filter_pd.src_desc() != input_x_mem.get_desc()) {
        conv_input_x_mem = memory(gradient_filter_pd.src_desc(), eng);
        reorder(input_x_mem, conv_input_x_mem)
            .execute(engine_stream, input_x_mem, conv_input_x_mem);
    }
    if (gradient_filter_pd.diff_weights_desc() != gradient_f_mem.get_desc()) {
        conv_gradient_f_mem =
            memory(gradient_filter_pd.diff_weights_desc(), eng);
        reorder(gradient_f_mem, conv_gradient_f_mem)
            .execute(engine_stream, gradient_f_mem, conv_gradient_f_mem);
    }
    if (gradient_filter_pd.diff_dst_desc() != gradient_y_mem.get_desc()) {
        conv_gradient_y_mem = memory(gradient_filter_pd.diff_dst_desc(), eng);
        reorder(gradient_y_mem, conv_gradient_y_mem)
            .execute(engine_stream, gradient_y_mem, conv_gradient_y_mem);
    }
    auto gradient_filter = convolution_backward_weights(gradient_filter_pd);

    gradient_filter.execute(engine_stream,
                            {{DNNL_ARG_SRC, conv_input_x_mem},
                             {DNNL_ARG_DIFF_WEIGHTS, conv_gradient_f_mem},
                             {DNNL_ARG_DIFF_DST, conv_gradient_y_mem}});
    if (gradient_filter_pd.diff_weights_desc() != gradient_f_mem.get_desc())
        reorder(conv_gradient_f_mem, gradient_f_mem)
            .execute(engine_stream, conv_gradient_f_mem, gradient_f_mem);
    else
        gradient_f_mem = conv_gradient_f_mem;
    engine_stream.wait();

    return 0;
}
