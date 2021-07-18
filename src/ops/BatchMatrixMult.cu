#include "gpu_runtime.h"

int DLGpuBatchMatrixMultiply(const DLArrayHandle matA, bool transposeA,
                             const DLArrayHandle matB, bool transposeB,
                             DLArrayHandle matC,
                             DLStreamHandle stream_handle = NULL) {
    assert(matA->ndim == matB->ndim);
    assert(matA->ndim == matC->ndim);

    int dev_id = (matA->ctx).device_id;
    cublas_init(dev_id, stream_handle);

    float one = 1.0f;
    float zero = 0.0f;

    int ndim = matA->ndim;
    int m = matC->shape[ndim - 1];
    int n = matC->shape[ndim - 2];
    int k = transposeA ? matA->shape[ndim - 2] : matA->shape[ndim - 1];
    long long int strideA = matA->shape[ndim - 2] * matA->shape[ndim - 1];
    long long int strideB = matB->shape[ndim - 2] * matB->shape[ndim - 1];
    long long int strideC = matC->shape[ndim - 2] * matC->shape[ndim - 1];

    int batchCount = 1;
    for (int i = 0; i < ndim - 2; ++i) {
        assert(matA->shape[i] == matB->shape[i]);
        assert(matA->shape[i] == matC->shape[i]);
        batchCount *= matA->shape[i];
    }

    cublasStatus_t res = cublasSgemmStridedBatched(
        cublas_map[dev_id], transposeB ? CUBLAS_OP_T : CUBLAS_OP_N,
        transposeA ? CUBLAS_OP_T : CUBLAS_OP_N, m, n, k, &one,
        (const float *)matB->data, !transposeB ? m : k, strideB,
        (const float *)matA->data, !transposeA ? k : n, strideA, &zero,
        (float *)matC->data, m, strideC, batchCount);
    assert(res == CUBLAS_STATUS_SUCCESS);
    return 0;
}
