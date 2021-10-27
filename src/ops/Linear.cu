#include "gpu_runtime.h"

__global__ void broadcast_linear_bias(const float *input_data, float *output_data,
    size_t input_size, size_t output_size) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= output_size)
    return;
    output_data[id] = input_data[id % input_size];
}

int DLGpuLinear(const DLArrayHandle matA, bool transposeA,
                const DLArrayHandle matB, bool transposeB,
                const DLArrayHandle bias,
                DLArrayHandle matC,
                DLStreamHandle stream_handle = NULL) {
    // cublas assume matrix is column major
    assert(matA->ndim == 2);
    assert(matB->ndim == 2);
    assert(bias->ndim == 1);
    assert(matC->ndim == 2);

    size_t input_size = bias->shape[0];
    size_t size = input_size * matC->shape[0];
    dim3 blocks;
    dim3 threads;
    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }
    if (stream_handle) {
        cudaStream_t *s = (cudaStream_t *)(stream_handle->handle);
        broadcast_linear_bias<<<blocks, threads, 0, *s>>>(
            (const float *)(bias->data), (float *)(matC->data), input_size, size);
    } else {
        broadcast_linear_bias<<<blocks, threads>>>(
            (const float *)(bias->data), (float *)(matC->data), input_size, size);
    }

    int dev_id = (matA->ctx).device_id;
    cublas_init(dev_id, stream_handle);

    float one = 1.0f;
    int m = matC->shape[1];
    int n = matC->shape[0];
    int k = transposeA ? matA->shape[0] : matA->shape[1];

    cublasSgemm(cublas_map[dev_id], transposeB ? CUBLAS_OP_T : CUBLAS_OP_N,
                transposeA ? CUBLAS_OP_T : CUBLAS_OP_N, m, n, k, &one,
                (const float *)matB->data, !transposeB ? m : k,
                (const float *)matA->data, !transposeA ? k : n, &one,
                (float *)matC->data, m);
    return 0;
}
