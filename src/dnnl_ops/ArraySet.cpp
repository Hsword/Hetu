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
#include <omp.h>

#include "dnnl.hpp"

#include "../common/c_runtime_api.h"
#include "dnnl_runtime.h"

using namespace dnnl;
using namespace std;

extern "C" int cpu_ArraySet(DLArrayHandle input, float value) {
    int num = 1;
    for (int i = 0; i < input->ndim; i++)
        num *= input->shape[i];
    float *data = (float *)(input->data);
#pragma omp parallel for
    for (int i = 0; i < num; i++)
        data[i] = value;
    return 0;
}
