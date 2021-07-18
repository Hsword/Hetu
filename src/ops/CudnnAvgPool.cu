#include "gpu_runtime.h"

int CuDNN_DLGpuAvgerage_Pooling2d(const DLArrayHandle input,
                                  const size_t kernel_H, const size_t kernel_W,
                                  DLArrayHandle output, const size_t padding,
                                  const size_t stride,
                                  DLStreamHandle stream_handle = NULL) {
    // create handle
    int dev_id = (input->ctx).device_id;
    cudnn_init(dev_id, stream_handle);

    // input
    size_t input_N = input->shape[0];
    size_t input_C = input->shape[1];
    size_t input_H = input->shape[2];
    size_t input_W = input->shape[3];
    const float *input_data = (const float *)input->data;

    // output
    size_t output_H = output->shape[2];
    size_t output_W = output->shape[3];
    float *output_data = (float *)output->data;

    // pooling descriptor
    cudnnPoolingDescriptor_t avgpool_desc;
    CUDNN_CALL(cudnnCreatePoolingDescriptor(&avgpool_desc));
    CUDNN_CALL(cudnnSetPooling2dDescriptor(
        avgpool_desc, CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
        CUDNN_PROPAGATE_NAN, kernel_H, kernel_W, padding, padding, stride,
        stride));

    // input descriptor
    cudnnTensorDescriptor_t input_desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT, input_N, input_C,
                                          input_H, input_W));

    // output descriptor
    cudnnTensorDescriptor_t output_desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT, input_N, input_C,
                                          output_H, output_W));

    float alpha = 1.0f;
    float beta = 0.0f;

    CUDNN_CALL(cudnnPoolingForward(cudnn_map[dev_id], avgpool_desc, &alpha,
                                   input_desc, input_data, &beta, output_desc,
                                   output_data));

    CUDNN_CALL(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_CALL(cudnnDestroyPoolingDescriptor(avgpool_desc));
    return 0;
}

int CuDNN_DLGpuAvgerage_Pooling2d_gradient(
    const DLArrayHandle output_Y, const DLArrayHandle gradient_Y,
    const DLArrayHandle input_X, const size_t kernel_H, const size_t kernel_W,
    DLArrayHandle gradient_X, const size_t padding, const size_t stride,
    DLStreamHandle stream_handle = NULL) {
    // create handle
    int dev_id = (input_X->ctx).device_id;
    cudnn_init(dev_id, stream_handle);

    // input
    size_t input_N = input_X->shape[0];
    size_t input_C = input_X->shape[1];
    size_t input_H = input_X->shape[2];
    size_t input_W = input_X->shape[3];
    const float *input_data = (const float *)input_X->data;
    float *gradient_x_data = (float *)gradient_X->data;
    // output
    size_t output_H = output_Y->shape[2];
    size_t output_W = output_Y->shape[3];
    const float *output_data = (const float *)output_Y->data;
    const float *gradient_Y_data = (const float *)gradient_Y->data;

    // pooling descriptor
    cudnnPoolingDescriptor_t avgpool_desc;
    CUDNN_CALL(cudnnCreatePoolingDescriptor(&avgpool_desc));
    CUDNN_CALL(cudnnSetPooling2dDescriptor(
        avgpool_desc, CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
        CUDNN_PROPAGATE_NAN, kernel_H, kernel_W, padding, padding, stride,
        stride));

    // input descriptor
    cudnnTensorDescriptor_t input_desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT, input_N, input_C,
                                          input_H, input_W));

    // output descriptor
    cudnnTensorDescriptor_t output_desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT, input_N, input_C,
                                          output_H, output_W));

    float alpha = 1.0f;
    float beta = 0.0f;

    CUDNN_CALL(cudnnPoolingBackward(cudnn_map[dev_id], avgpool_desc, &alpha,
                                    output_desc, output_data, output_desc,
                                    gradient_Y_data, input_desc, input_data,
                                    &beta, input_desc, gradient_x_data));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_CALL(cudnnDestroyPoolingDescriptor(avgpool_desc));
    return 0;
}
