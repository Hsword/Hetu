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

extern "C" int DnnlMatrixMultiply(const DLArrayHandle matA, bool transposeA,
                                  const DLArrayHandle matB, bool transposeB,
                                  DLArrayHandle matC) {
    dnnl_stream_init();

    assert(matA->ndim == 2 && matB->ndim == 2 && matC->ndim == 2);
    memory::desc srcA_md, srcB_md, dst_md;
    if (!transposeA)
        srcA_md = memory::desc({matA->shape[0], matA->shape[1]},
                               memory::data_type::f32, memory::format_tag::ab);
    else
        srcA_md = memory::desc({matA->shape[1], matA->shape[0]},
                               memory::data_type::f32, memory::format_tag::ba);
    if (!transposeB)
        srcB_md = memory::desc({matB->shape[0], matB->shape[1]},
                               memory::data_type::f32, memory::format_tag::ab);
    else
        srcB_md = memory::desc({matB->shape[1], matB->shape[0]},
                               memory::data_type::f32, memory::format_tag::ba);
    dst_md = memory::desc({matC->shape[0], matC->shape[1]},
                          memory::data_type::f32, memory::format_tag::ab);
    ;

    auto srcA_mem = memory(srcA_md, eng, matA->data);
    auto srcB_mem = memory(srcB_md, eng, matB->data);
    auto dst_mem = memory(dst_md, eng, matC->data);

    auto Matmul_d = matmul::desc(srcA_md, srcB_md, dst_md);
    auto Matmul_pd = matmul::primitive_desc(Matmul_d, eng);
    auto Matmul = matmul(Matmul_pd);

    Matmul.execute(engine_stream, {{DNNL_ARG_SRC, srcA_mem},
                                   {DNNL_ARG_WEIGHTS, srcB_mem},
                                   {DNNL_ARG_DST, dst_mem}});

    engine_stream.wait();
    return 0;
}