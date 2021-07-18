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

// DLArrayHandle A,B,C,D;

extern "C" int DnnlSoftmax(const DLArrayHandle input, DLArrayHandle output) {
    // engine eng(engine::kind::cpu, 0);
    // stream engine_stream(eng);
    dnnl_stream_init();

    assert(input->ndim == 2);
    auto mat_md = memory::desc({input->shape[0], input->shape[1]},
                               memory::data_type::f32, memory::format_tag::ab);
    auto src_mem = memory(mat_md, eng, input->data);
    auto dst_softmax_mem = memory(mat_md, eng, output->data);

    auto softmax_d =
        softmax_forward::desc(prop_kind::forward_training, mat_md, 1);
    auto softmax_pd = softmax_forward::primitive_desc(softmax_d, eng);
    auto softmax = softmax_forward(softmax_pd);

    softmax.execute(engine_stream,
                    {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_DST, dst_softmax_mem}});
    engine_stream.wait();
    return 0;
}
