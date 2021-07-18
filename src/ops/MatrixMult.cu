#include "gpu_runtime.h"

cublasHandle_t cublas_handle = NULL;
int DLGpuMatrixMultiply(const DLArrayHandle matA, bool transposeA,
                        const DLArrayHandle matB, bool transposeB,
                        DLArrayHandle matC,
                        DLStreamHandle stream_handle = NULL) {
    // cublas assume matrix is column major
    assert(matA->ndim == 2);
    assert(matB->ndim == 2);
    assert(matC->ndim == 2);

    int dev_id = (matA->ctx).device_id;
    cublas_init(dev_id, stream_handle);

    float one = 1.0f;
    float zero = 0.0f;
    int m = matC->shape[1];
    int n = matC->shape[0];
    int k = transposeA ? matA->shape[0] : matA->shape[1];

    cublasSgemm(cublas_map[dev_id], transposeB ? CUBLAS_OP_T : CUBLAS_OP_N,
                transposeA ? CUBLAS_OP_T : CUBLAS_OP_N, m, n, k, &one,
                (const float *)matB->data, !transposeB ? m : k,
                (const float *)matA->data, !transposeA ? k : n, &zero,
                (float *)matC->data, m);
    return 0;
}
