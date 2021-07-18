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

extern "C" int DnnlMatrixElementwiseDivide(const DLArrayHandle matA,
                                           const DLArrayHandle matB,
                                           DLArrayHandle output) {
    // engine eng(engine::kind::cpu, 0);
    // stream engine_stream(eng);
    dnnl_stream_init();

    vector<long int> shape, format;
    for (int i = 0; i < matA->ndim; i++)
        shape.push_back(matA->shape[i]);
    format.resize(matA->ndim);
    format[(matA->ndim) - 1] = 1;
    for (int i = format.size() - 2; i >= 0; i--)
        format[i] = format[i + 1] * shape[i + 1];
    float *temp = new float[shape[0] * format[0]];
    auto mat_md = memory::desc(shape, memory::data_type::f32, format);
    auto srcA_mem = memory(mat_md, eng, matA->data);
    auto srcB_mem = memory(mat_md, eng, matB->data);
    auto temp_mem = memory(mat_md, eng, temp);
    auto dst_mem = memory(mat_md, eng, output->data);

    auto Reciprocal_d = eltwise_forward::desc(
        prop_kind::forward_training, algorithm::eltwise_pow, mat_md, 1.f, -1.f);
    auto Reciprocal_pd = eltwise_forward::primitive_desc(Reciprocal_d, eng);
    auto Reciprocal = eltwise_forward(Reciprocal_pd);
    auto MultiplyElewise_d =
        binary::desc(algorithm::binary_mul, mat_md, mat_md, mat_md);
    auto MultiplyElewise_pd = binary::primitive_desc(MultiplyElewise_d, eng);
    auto MultiplyElewise = binary(MultiplyElewise_pd);

    Reciprocal.execute(engine_stream,
                       {{DNNL_ARG_SRC, srcB_mem}, {DNNL_ARG_DST, temp_mem}});
    engine_stream.wait();
    MultiplyElewise.execute(engine_stream, {{DNNL_ARG_SRC_0, srcA_mem},
                                            {DNNL_ARG_SRC_1, temp_mem},
                                            {DNNL_ARG_DST, dst_mem}});
    engine_stream.wait();
    delete temp;
    return 0;
}
