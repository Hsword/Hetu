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

extern "C" int DnnlOpposite(const DLArrayHandle input, DLArrayHandle output) {
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

    auto Opposite_d =
        eltwise_forward::desc(prop_kind::forward_training,
                              algorithm::eltwise_linear, mat_md, -1.f, 0.f);
    auto Opposite_pd = eltwise_forward::primitive_desc(Opposite_d, eng);
    auto Opposite = eltwise_forward(Opposite_pd);

    Opposite.execute(engine_stream,
                     {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_DST, dst_mem}});
    engine_stream.wait();
    return 0;
}
