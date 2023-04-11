#include "gpu_runtime.h"

int CuDNN_DLGpuSoftmax(const DLArrayHandle input, DLArrayHandle output,
                       DLStreamHandle stream_handle = NULL) {
    size_t indim = input->ndim;
    assert(indim == output->ndim);
    int dev_id = (input->ctx).device_id;
    cudnn_init(dev_id, stream_handle);
    int n_ = 1;
    for (int i = 0; i < indim - 1; ++i) {
        n_ *= input->shape[i];
    }
    int c_ = input->shape[indim - 1];
    const float *input_data = (const float *)(input->data);
    float *output_data = (float *)(output->data);
    float alpha = 1.0f;
    float beta = 0.0f;
    cudnnTensorDescriptor_t desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&desc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT, n_, c_, 1, 1));
    CUDNN_CALL(cudnnSoftmaxForward(cudnn_map[dev_id], CUDNN_SOFTMAX_ACCURATE,
                                   CUDNN_SOFTMAX_MODE_INSTANCE, &alpha, desc,
                                   (const void *)input_data, &beta, desc,
                                   (void *)output_data));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(desc));
    return 0;
}

int CuDNN_DLGpuSoftmaxGradient(const DLArrayHandle y_arr,
                               const DLArrayHandle dy, DLArrayHandle dx,
                               DLStreamHandle stream_handle = NULL) {
    size_t indim = y_arr->ndim;
    assert(indim == dy->ndim && indim == dx->ndim);
    int dev_id = (y_arr->ctx).device_id;
    cudnn_init(dev_id, stream_handle);
    int n_ = 1;
    for (int i = 0; i < indim - 1; ++i) {
        n_ *= y_arr->shape[i];
    }
    int c_ = y_arr->shape[indim - 1];
    const float *y_data = (const float *)(y_arr->data);
    const float *dy_data = (const float *)(dy->data);
    float *dx_data = (float *)(dx->data);
    float alpha = 1.0f;
    float beta = 0.0f;
    cudnnTensorDescriptor_t desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&desc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT, n_, c_, 1, 1));
    CUDNN_CALL(cudnnSoftmaxBackward(
        cudnn_map[dev_id], CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
        &alpha, desc, (const void *)y_data, desc, (const void *)dy_data, &beta,
        desc, (void *)dx_data));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(desc));
    return 0;
}

int CuDNN_DLGpuLogSoftmax(const DLArrayHandle input, DLArrayHandle output,
                          DLStreamHandle stream_handle = NULL) {
    size_t indim = input->ndim;
    assert(indim == output->ndim);
    int dev_id = (input->ctx).device_id;
    cudnn_init(dev_id, stream_handle);
    int n_ = 1;
    for (int i = 0; i < indim - 1; ++i) {
        n_ *= input->shape[i];
    }
    int c_ = input->shape[indim - 1];
    const float *input_data = (const float *)(input->data);
    float *output_data = (float *)(output->data);
    float alpha = 1.0f;
    float beta = 0.0f;
    cudnnTensorDescriptor_t desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&desc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT, n_, c_, 1, 1));
    CUDNN_CALL(cudnnSoftmaxForward(cudnn_map[dev_id], CUDNN_SOFTMAX_LOG,
                                   CUDNN_SOFTMAX_MODE_INSTANCE, &alpha, desc,
                                   (const void *)input_data, &beta, desc,
                                   (void *)output_data));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(desc));
    return 0;
}

int CuDNN_DLGpuLogSoftmaxGradient(const DLArrayHandle y_arr,
                                  const DLArrayHandle dy, DLArrayHandle dx,
                                  DLStreamHandle stream_handle = NULL) {
    size_t indim = y_arr->ndim;
    assert(indim == dy->ndim && indim == dx->ndim);
    int dev_id = (y_arr->ctx).device_id;
    cudnn_init(dev_id, stream_handle);
    int n_ = 1;
    for (int i = 0; i < indim - 1; ++i) {
        n_ *= y_arr->shape[i];
    }
    int c_ = y_arr->shape[indim - 1];
    const float *y_data = (const float *)(y_arr->data);
    const float *dy_data = (const float *)(dy->data);
    float *dx_data = (float *)(dx->data);
    float alpha = 1.0f;
    float beta = 0.0f;
    cudnnTensorDescriptor_t desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&desc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT, n_, c_, 1, 1));
    CUDNN_CALL(cudnnSoftmaxBackward(
        cudnn_map[dev_id], CUDNN_SOFTMAX_LOG, CUDNN_SOFTMAX_MODE_INSTANCE,
        &alpha, desc, (const void *)y_data, desc, (const void *)dy_data, &beta,
        desc, (void *)dx_data));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(desc));
    return 0;
}
