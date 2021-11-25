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

extern "C" int DnnlGelu(const DLArrayHandle input, DLArrayHandle output) {
    printf("DnnlGelu is not implemented yet.\n");
    return 0;
}

extern "C" int DnnlGelu_Gradient(const DLArrayHandle input,
                                 const DLArrayHandle in_grad,
                                 DLArrayHandle output) {
    printf("DnnlGelu_Gradient is not implemented yet.\n");
    return 0;
}