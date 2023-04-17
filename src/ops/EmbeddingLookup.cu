#include "gpu_runtime.h"
#include "dispatch.h"

template <typename spec_t>
__global__ void embedding_lookup_kernel(const spec_t *input, const int *ids,
                                        spec_t *output, size_t nrow,
                                        size_t length, size_t size) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size)
        return;
    int id = ids[index];
    spec_t *output_ptr = output + length * index;
    if (id < 0 || id >= nrow) {
        for (int i = 0; i < length; ++i)
            output_ptr[i] = 0;
    } else {
        const spec_t *input_ptr = input + length * id;
        for (int i = 0; i < length; ++i)
            output_ptr[i] = input_ptr[i];
    }
}

int DLGpuEmbeddingLookUp(const DLArrayHandle input, const DLArrayHandle ids,
                         DLArrayHandle output,
                         DLStreamHandle stream_handle = NULL) {
    assert(input->ndim == 2);
    size_t size = ArrSize(ids);
    for (int i = 0; i < output->ndim; i++) {
        if (i < output->ndim - 1) {
            assert(ids->shape[i] == output->shape[i]);
        } else {
            assert(input->shape[1] == output->shape[i]);
        }
    }
    size_t nrow = input->shape[0], length = input->shape[1];
    dim3 blocks;
    dim3 threads;
    ThreadBlock1D(threads, blocks, size);
    void *output_data = output->data;
    void *input_data = input->data;
    const int *id_list = (const int *)ids->data;
    assert(stream_handle != NULL);
    cudaStream_t stream = *(cudaStream_t *)stream_handle->handle;
    HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(input->dtype, spec_t, [&]() {
        embedding_lookup_kernel<spec_t><<<blocks, threads, 0, stream>>>(
            (const spec_t *)input_data, id_list, (spec_t *)output_data, nrow,
            length, size);
    });
    return 0;
}
