#include "gpu_runtime.h"

int DLGpuAddmm(const DLArrayHandle input, const DLArrayHandle matA,
               const DLArrayHandle matB, float alpha, float beta,
               DLArrayHandle matC, DLStreamHandle stream_handle = NULL) {
    // cublas assume matrix is column major
    assert(input->ndim == 2);
    assert(matA->ndim == 2);
    assert(matB->ndim == 2);
    assert(matC->ndim == 2);

    int dev_id = (input->ctx).device_id;
    cublas_init(dev_id, stream_handle);

    int m = matC->shape[1];
    int n = matC->shape[0];
    int k = matA->shape[1];

    float *input_data = (float *)input->data;
    float *output_data = (float *)matC->data;
    int size = 1;
    for (int i = 0; i < input->ndim; i++) {
        size *= input->shape[i];
    }

    cudaMemcpy((void *)output_data, (void *)input_data, size * sizeof(float),
               cudaMemcpyDeviceToDevice);

    cublasSgemm(cublas_map[dev_id], CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
                (const float *)matB->data, m, (const float *)matA->data, k,
                &beta, (float *)output_data, m);
    return 0;
}

int DLGpuAddmmGradient(const DLArrayHandle input, DLArrayHandle output,
                       int axis, float beta,
                       DLStreamHandle stream_handle = NULL) {
    int dev_id = (input->ctx).device_id;
    cudnn_init(dev_id, stream_handle);

    float zero = 0.0f;
    cudnnReduceTensorDescriptor_t rtd;
    CUDNN_CALL(cudnnCreateReduceTensorDescriptor(&rtd));
    CUDNN_CALL(cudnnSetReduceTensorDescriptor(
        rtd, CUDNN_REDUCE_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN,
        CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES));

    cudnnTensorDescriptor_t adesc;
    cudnnTensorDescriptor_t cdesc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&adesc));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&cdesc));

    int ori_ndim = input->ndim;
    int ndim = max(ori_ndim, 4);
    size_t cpu_mem = ndim * sizeof(int);
    int *dimA = (int *)malloc(cpu_mem);
    int *strideA = (int *)malloc(cpu_mem);
    int *dimC = (int *)malloc(cpu_mem);
    int *strideC = (int *)malloc(cpu_mem);

    for (int i = 0; i < ori_ndim; ++i) {
        dimA[i] = dimC[i] = (int)input->shape[i];
    }
    for (int i = ori_ndim; i < ndim; ++i) {
        dimA[i] = dimC[i] = 1;
    }
    dimC[0] = 1;
    int temp_strideA = 1;
    int temp_strideC = 1;
    for (int i = ndim - 1; i >= 0; --i) {
        strideA[i] = temp_strideA;
        strideC[i] = temp_strideC;
        temp_strideA *= dimA[i];
        temp_strideC *= dimC[i];
    }

    size_t size = temp_strideA * sizeof(float);

    if (is_chunk_init(dev_id) == false) {
        chunk_init(dev_id);
    }
    void *workspace = find_chunk(size, dev_id);

    CUDNN_CALL(cudnnSetTensorNdDescriptor(adesc, CUDNN_DATA_FLOAT, ndim, dimA,
                                          strideA));
    CUDNN_CALL(cudnnSetTensorNdDescriptor(cdesc, CUDNN_DATA_FLOAT, ndim, dimC,
                                          strideC));
    CUDNN_CALL(cudnnReduceTensor(cudnn_map[dev_id], rtd, NULL, 0, workspace,
                                 size, &beta, adesc, (const void *)input->data,
                                 &zero, cdesc, (void *)output->data));

    del_chunk(workspace, dev_id);
    CUDNN_CALL(cudnnDestroyTensorDescriptor(adesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(cdesc));
    CUDNN_CALL(cudnnDestroyReduceTensorDescriptor(rtd));
    free(dimA);
    free(dimC);
    free(strideA);
    free(strideC);
    return 0;
}
