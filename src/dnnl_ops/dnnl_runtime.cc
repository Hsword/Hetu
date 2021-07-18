#include "./dnnl_runtime.h"

bool is_dnnl_stream_init = 0;
engine eng;
stream engine_stream;

void dnnl_stream_init() {
    if (is_dnnl_stream_init == 0) {
        engine eng1(engine::kind::cpu, 0);
        eng = eng1;
        stream engine_stream1(eng);
        engine_stream = engine_stream1;
        is_dnnl_stream_init = 1;
    }
}
void print_dlarray(DLArrayHandle mat) {
    const float *mat_data = (const float *)mat->data;
    size_t input_size = 1;
    for (index_t i = 0; i < mat->ndim; i++) {
        input_size *= mat->shape[i];
    }
    for (size_t i = 0; i < input_size; i++) {
        std::cout << mat_data[i] << ' ';
    }
    std::cout << std::endl;
}

void read_from_dnnl_memory(void *handle, dnnl::memory &mem) {
    dnnl::engine eng = mem.get_engine();
    size_t bytes = mem.get_desc().get_size();

    if (eng.get_kind() == dnnl::engine::kind::cpu) {
        uint8_t *src = static_cast<uint8_t *>(mem.get_data_handle());
        for (size_t i = 0; i < bytes; ++i)
            ((uint8_t *)handle)[i] = src[i];
    }
}
