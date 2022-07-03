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

int DnnlBatchNorm(const DLArrayHandle input, const DLArrayHandle bn_scale,
                  const DLArrayHandle bn_bias, DLArrayHandle output,
                  DLArrayHandle running_mean, DLArrayHandle running_var,
                  DLArrayHandle save_mean, DLArrayHandle save_var,
                  float momentum, float eps) {
    dnnl_stream_init();

    assert(input->ndim == 4);

    int N = input->shape[0];
    int C = input->shape[1];
    int H = input->shape[2];
    int W = input->shape[3];
    assert(N == output->shape[0]);
    assert(C == output->shape[1]);
    assert(H == output->shape[2]);
    assert(W == output->shape[3]);

    float *scale = (float *)(bn_scale->data);
    float *bias = (float *)(bn_bias->data);
    float *running_mean_data = (float *)(running_mean->data);
    float *running_var_data = (float *)(running_var->data);
    float *save_mean_data = (float *)(save_mean->data);
    float *save_var_data = (float *)(save_var->data);
    float *ptr = new float[2 * C];
    for (int i = 0; i < C; i++)
        ptr[i] = scale[i];
    for (int i = 0; i < C; i++)
        ptr[i + C] = bias[i];

    auto data_md = memory::desc({N, C, H, W}, memory::data_type::f32,
                                memory::format_tag::nchw);
    auto ptr_md =
        memory::desc({2, C}, memory::data_type::f32, memory::format_tag::ab);
    auto mean_var_md =
        memory::desc({C}, memory::data_type::f32, memory::format_tag::a);

    auto input_mem = memory(data_md, eng, input->data);
    auto output_mem = memory(data_md, eng, output->data);
    auto ptr_mem = memory(ptr_md, eng, ptr);
    auto mean_mem = memory(mean_var_md, eng, save_mean_data);
    auto var_mem = memory(mean_var_md, eng, save_var_data);

    auto bn_d = batch_normalization_forward::desc(
        prop_kind::forward_training, data_md, eps,
        normalization_flags::use_scale_shift);
    auto bn_pd = batch_normalization_forward::primitive_desc(bn_d, eng);
    auto bn = batch_normalization_forward(bn_pd);

    bn.execute(engine_stream, {{DNNL_ARG_SRC, input_mem},
                               {DNNL_ARG_SCALE_SHIFT, ptr_mem},
                               {DNNL_ARG_MEAN, mean_mem},
                               {DNNL_ARG_VARIANCE, var_mem},
                               {DNNL_ARG_DST, output_mem}});
    engine_stream.wait();
    for (int i = 0; i < C; ++i) {
        running_mean_data[i] = running_mean_data[i] * (1 - momentum)
                               + save_mean_data[i] * momentum;
        running_var_data[i] =
            running_var_data[i] * (1 - momentum) + save_var_data[i] * momentum;
    }

    return 0;
}

int DnnlBatchNorm_Gradient(const DLArrayHandle grad_y,
                           const DLArrayHandle input,
                           const DLArrayHandle bn_scale, DLArrayHandle grad_x,
                           DLArrayHandle grad_scale, DLArrayHandle grad_bias,
                           DLArrayHandle mean, DLArrayHandle var,
                           const float eps) {
    dnnl_stream_init();

    assert(input->ndim == 4);

    int N = input->shape[0];
    int C = input->shape[1];
    int H = input->shape[2];
    int W = input->shape[3];
    int out_N = grad_y->shape[0];
    int out_C = grad_y->shape[1];
    int out_H = grad_y->shape[2];
    int out_W = grad_y->shape[3];

    float *scale = (float *)(bn_scale->data);
    float *ptr = new float[2 * C];
    float *grad_ptr = new float[2 * out_C];
    for (int i = 0; i < C; i++)
        ptr[i] = scale[i];
    // bn_bias is not used in gradient
    // tests show it's ok to set here 0
    for (int i = 0; i < C; i++)
        ptr[i + C] = 0;

    auto data_md = memory::desc({N, C, H, W}, memory::data_type::f32,
                                memory::format_tag::nchw);
    auto diff_md =
        memory::desc({out_N, out_C, out_H, out_W}, memory::data_type::f32,
                     memory::format_tag::nchw);
    auto ptr_md =
        memory::desc({2, C}, memory::data_type::f32, memory::format_tag::ab);
    auto grad_ptr_md = memory::desc({2, out_C}, memory::data_type::f32,
                                    memory::format_tag::ab);
    auto mean_var_md =
        memory::desc({C}, memory::data_type::f32, memory::format_tag::a);

    auto input_mem = memory(data_md, eng, input->data);
    auto grad_x_mem = memory(diff_md, eng, grad_x->data);
    auto grad_y_mem = memory(diff_md, eng, grad_y->data);
    auto ptr_mem = memory(ptr_md, eng, ptr);
    auto grad_ptr_mem = memory(grad_ptr_md, eng, grad_ptr);
    auto mean_mem = memory(mean_var_md, eng, mean->data);
    auto var_mem = memory(mean_var_md, eng, var->data);

    auto bn_d = batch_normalization_forward::desc(
        prop_kind::forward_training, data_md, eps,
        normalization_flags::use_scale_shift);
    auto bn_pd = batch_normalization_forward::primitive_desc(bn_d, eng);
    auto bn_grad_d = batch_normalization_backward::desc(
        prop_kind::backward, diff_md, data_md, eps,
        normalization_flags::use_scale_shift);
    auto bn_grad_pd =
        batch_normalization_backward::primitive_desc(bn_grad_d, eng, bn_pd);
    auto bn_grad = batch_normalization_backward(bn_grad_pd);

    bn_grad.execute(engine_stream, {{DNNL_ARG_SRC, input_mem},
                                    {DNNL_ARG_SCALE_SHIFT, ptr_mem},
                                    {DNNL_ARG_MEAN, mean_mem},
                                    {DNNL_ARG_VARIANCE, var_mem},
                                    {DNNL_ARG_DIFF_DST, grad_y_mem},
                                    {DNNL_ARG_DIFF_SRC, grad_x_mem},
                                    {DNNL_ARG_DIFF_SCALE_SHIFT, grad_ptr_mem}});
    engine_stream.wait();

    scale = (float *)(grad_scale->data);
    float *bias = (float *)(grad_bias->data);
    for (int i = 0; i < out_C; i++)
        scale[i] = grad_ptr[i];
    for (int i = 0; i < out_C; i++)
        bias[i] = grad_ptr[i + C];
    return 0;
}

int DnnlBatchNorm_Inference(const DLArrayHandle input,
                            const DLArrayHandle bn_scale,
                            const DLArrayHandle bn_bias, DLArrayHandle output,
                            DLArrayHandle mean, DLArrayHandle var, float eps) {
    dnnl_stream_init();

    assert(input->ndim == 4);

    int N = input->shape[0];
    int C = input->shape[1];
    int H = input->shape[2];
    int W = input->shape[3];
    assert(N == output->shape[0]);
    assert(C == output->shape[1]);
    assert(H == output->shape[2]);
    assert(W == output->shape[3]);

    float *scale = (float *)(bn_scale->data);
    float *bias = (float *)(bn_bias->data);
    float *ptr = new float[2 * C];
    float *running_mean = (float *)mean->data;
    float *running_var = (float *)var->data;
    for (int i = 0; i < C; i++)
        ptr[i] = scale[i];
    for (int i = 0; i < C; i++)
        ptr[i + C] = bias[i];

    auto data_md = memory::desc({N, C, H, W}, memory::data_type::f32,
                                memory::format_tag::nchw);
    auto ptr_md =
        memory::desc({2, C}, memory::data_type::f32, memory::format_tag::ab);
    auto mean_var_md =
        memory::desc({C}, memory::data_type::f32, memory::format_tag::a);

    auto input_mem = memory(data_md, eng, input->data);
    auto output_mem = memory(data_md, eng, output->data);
    auto ptr_mem = memory(ptr_md, eng, ptr);
    auto mean_mem = memory(mean_var_md, eng, running_mean);
    auto var_mem = memory(mean_var_md, eng, running_var);

    auto bn_d = batch_normalization_forward::desc(
        prop_kind::forward_inference, data_md, eps,
        normalization_flags::use_global_stats
            | normalization_flags::use_scale_shift);
    auto bn_pd = batch_normalization_forward::primitive_desc(bn_d, eng);
    auto bn = batch_normalization_forward(bn_pd);

    bn.execute(engine_stream, {{DNNL_ARG_SRC, input_mem},
                               {DNNL_ARG_SCALE_SHIFT, ptr_mem},
                               {DNNL_ARG_MEAN, mean_mem},
                               {DNNL_ARG_VARIANCE, var_mem},
                               {DNNL_ARG_DST, output_mem}});
    engine_stream.wait();

    return 0;
}