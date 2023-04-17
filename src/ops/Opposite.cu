#include "gpu_runtime.h"
#include "dispatch.h"

template <typename spec_t>
__global__ void opposite_kernel(const spec_t *input, spec_t *output,
                                size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    output[ind] = -input[ind];
}

int DLGpuOpposite(const DLArrayHandle input, DLArrayHandle output,
                  DLStreamHandle stream_handle = NULL) {
    size_t size = ArrSize(input);
    dim3 blocks;
    dim3 threads;
    ThreadBlock1D(threads, blocks, size);
    void *input_data = input->data;
    void *output_data = output->data;
    assert(stream_handle != NULL);
    cudaStream_t stream = *(cudaStream_t *)stream_handle->handle;
    HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(input->dtype, spec_t, [&]() {
        opposite_kernel<spec_t><<<blocks, threads, 0, stream>>>(
            (const spec_t *)input_data, (spec_t *)output_data, size);
    });
    return 0;
}
