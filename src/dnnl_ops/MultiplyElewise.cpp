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

extern "C" int DnnlMatrixElementwiseMultiply(const DLArrayHandle matA,
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
    auto mat_md = memory::desc(shape, memory::data_type::f32, format);

    auto srcA_mem = memory(mat_md, eng, matA->data);
    auto srcB_mem = memory(mat_md, eng, matB->data);
    auto dst_mem = memory(mat_md, eng, output->data);
    auto MultiplyElewise_d =
        binary::desc(algorithm::binary_mul, mat_md, mat_md, mat_md);
    auto MultiplyElewise_pd = binary::primitive_desc(MultiplyElewise_d, eng);
    auto MultiplyElewise = binary(MultiplyElewise_pd);

    MultiplyElewise.execute(engine_stream, {{DNNL_ARG_SRC_0, srcA_mem},
                                            {DNNL_ARG_SRC_1, srcB_mem},
                                            {DNNL_ARG_DST, dst_mem}});
    engine_stream.wait();
    return 0;
}
