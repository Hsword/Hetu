#include "gpu_runtime.h"
#include "dispatch.h"

template <typename spec_t>
__global__ void general_memory_copy_kernel(spec_t *A, const spec_t *B,
                                           size_t len) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= len)
        return;
    A[id] = B[id];
}

int DLGpuReshape(const DLArrayHandle in_arr, DLArrayHandle out_arr,
                 DLStreamHandle stream_handle = NULL) {
    size_t input_size = ArrSize(in_arr);
    size_t output_size = ArrSize(out_arr);
    assert(input_size == output_size);
    void *input_data = in_arr->data;
    void *output_data = out_arr->data;
    dim3 threads, blocks;
    ThreadBlock1D(threads, blocks, input_size);
    assert(stream_handle != NULL);
    cudaStream_t stream = *(cudaStream_t *)stream_handle->handle;
    HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(in_arr->dtype, spec_t, [&]() {
        general_memory_copy_kernel<spec_t><<<blocks, threads, 0, stream>>>(
            (spec_t *)output_data, (const spec_t *)input_data, input_size);
    });
    return 0;
}
