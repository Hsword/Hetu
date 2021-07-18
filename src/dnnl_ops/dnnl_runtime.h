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

using namespace dnnl;

extern bool is_dnnl_stream_init;
extern engine eng;
extern stream engine_stream;

void dnnl_stream_init();
void print_dlarray(DLArrayHandle mat);
void read_from_dnnl_memory(void *handle, dnnl::memory &mem);
