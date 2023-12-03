#include "gpu_runtime.h"

// the shape of bn_scale/bias   1*C*1*1
int CuDNN_DLGpuBatch_Normalization(
    const DLArrayHandle input_X, const DLArrayHandle bn_scale,
    const DLArrayHandle bn_bias, DLArrayHandle output_Y, double momentum,
    double eps, DLArrayHandle running_mean, DLArrayHandle running_var,
    DLArrayHandle save_mean, DLArrayHandle save_var,
    DLStreamHandle stream_handle = NULL) {
    int dev_id = (input_X->ctx).device_id;
    cudaSetDevice(dev_id);
    cudnn_init(dev_id, stream_handle);

    // input
    size_t input_N, input_C, input_H, input_W;
    if (input_X->ndim == 4) {
        input_N = input_X->shape[0];
        input_C = input_X->shape[1];
        input_H = input_X->shape[2];
        input_W = input_X->shape[3];
    } else if (input_X->ndim == 3) {
        input_N = input_X->shape[0];
        input_C = input_X->shape[1];
        input_H = input_X->shape[2];
        input_W = 1;
    } else if (input_X->ndim == 2) {
        input_N = input_X->shape[0];
        input_C = input_X->shape[1];
        input_H = 1;
        input_W = 1;
    }
    const float *input_data = (const float *)(input_X->data);

    // input descriptor
    cudnnTensorDescriptor_t input_desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT, input_N, input_C,
                                          input_H, input_W));

    // output
    float *output_data = (float *)(output_Y->data);

    // output descriptor
    cudnnTensorDescriptor_t output_desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT, input_N, input_C,
                                          input_H, input_W));

    // bn parameter descriptor
    cudnnTensorDescriptor_t bnScaleBiasMeanVar_desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&bnScaleBiasMeanVar_desc));
    CUDNN_CALL(
        cudnnDeriveBNTensorDescriptor(bnScaleBiasMeanVar_desc, input_desc,
                                      CUDNN_BATCHNORM_SPATIAL)); // after conv

    // bn parameter
    const float *bn_scale_data = (const float *)(bn_scale->data);
    const float *bn_bias_data = (const float *)(bn_bias->data);

    void *running_mean_arr = running_mean->data;
    void *running_var_arr = running_var->data;
    void *save_mean_arr = save_mean->data;
    void *save_var_arr = save_var->data;
    float alpha = 1.0f;
    float beta = 0.0f;

    CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
        cudnn_map[dev_id], CUDNN_BATCHNORM_SPATIAL, &alpha, &beta, input_desc,
        input_data, output_desc, output_data, bnScaleBiasMeanVar_desc,
        bn_scale_data, bn_bias_data, momentum, running_mean_arr,
        running_var_arr, eps, save_mean_arr, save_var_arr));

    CUDNN_CALL(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(bnScaleBiasMeanVar_desc));

    return 0;
}

int CuDNN_DLGpuBatch_Normalization_gradient(
    const DLArrayHandle gradient_Y, const DLArrayHandle input_X,
    const DLArrayHandle bn_scale, DLArrayHandle gradient_X,
    DLArrayHandle gradient_bn_scale, DLArrayHandle gradient_bn_bias, double eps,
    DLArrayHandle save_mean, DLArrayHandle save_var,
    DLStreamHandle stream_handle = NULL) {
    int dev_id = (input_X->ctx).device_id;
    cudnn_init(dev_id, stream_handle);

    // input
    size_t input_N, input_C, input_H, input_W;
    if (input_X->ndim == 4) {
        input_N = input_X->shape[0];
        input_C = input_X->shape[1];
        input_H = input_X->shape[2];
        input_W = input_X->shape[3];
    } else if (input_X->ndim == 3) {
        input_N = input_X->shape[0];
        input_C = input_X->shape[1];
        input_H = input_X->shape[2];
        input_W = 1;
    } else if (input_X->ndim == 2) {
        input_N = input_X->shape[0];
        input_C = input_X->shape[1];
        input_H = 1;
        input_W = 1;
    }
    const float *input_data = (const float *)(input_X->data);

    // input descriptor
    cudnnTensorDescriptor_t input_desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT, input_N, input_C,
                                          input_H, input_W));

    // output
    const float *gradient_y_data = (const float *)(gradient_Y->data);

    // output descriptor
    cudnnTensorDescriptor_t output_desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT, input_N, input_C,
                                          input_H, input_W));

    // bn parameter descriptor
    cudnnTensorDescriptor_t bnScaleBiasMeanVar_desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&bnScaleBiasMeanVar_desc));
    CUDNN_CALL(
        cudnnDeriveBNTensorDescriptor(bnScaleBiasMeanVar_desc, input_desc,
                                      CUDNN_BATCHNORM_SPATIAL)); // after conv

    const float *bn_scale_data = (const float *)(bn_scale->data);

    // x gradient
    float *gradient_x_data = (float *)(gradient_X->data);
    // bn gradient
    float *gradient_bn_bias_data = (float *)(gradient_bn_bias->data);
    float *gradient_bn_scale_data = (float *)(gradient_bn_scale->data);
    void *save_mean_arr = save_mean->data;
    void *save_var_arr = save_var->data;
    float one = 1.0f;
    float zero = 0.0f;

    CUDNN_CALL(cudnnBatchNormalizationBackward(
        cudnn_map[dev_id], CUDNN_BATCHNORM_SPATIAL_PERSISTENT, &one, &zero,
        &one, &zero, input_desc, input_data, output_desc, gradient_y_data,
        input_desc, gradient_x_data, bnScaleBiasMeanVar_desc, bn_scale_data,
        gradient_bn_scale_data, gradient_bn_bias_data, eps, save_mean_arr,
        save_var_arr));

    CUDNN_CALL(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(bnScaleBiasMeanVar_desc));
    return 0;
}

int CuDNN_DLGpuBatch_Normalization_inference(
    const DLArrayHandle input_X, const DLArrayHandle bn_scale,
    const DLArrayHandle bn_bias, DLArrayHandle output_Y, double eps,
    DLArrayHandle estimated_mean, DLArrayHandle estimated_var,
    DLStreamHandle stream_handle = NULL) {
    int dev_id = (input_X->ctx).device_id;
    cudaSetDevice(dev_id);
    cudnn_init(dev_id, stream_handle);

    // input
    size_t input_N, input_C, input_H, input_W;
    if (input_X->ndim == 4) {
        input_N = input_X->shape[0];
        input_C = input_X->shape[1];
        input_H = input_X->shape[2];
        input_W = input_X->shape[3];
    } else if (input_X->ndim == 3) {
        input_N = input_X->shape[0];
        input_C = input_X->shape[1];
        input_H = input_X->shape[2];
        input_W = 1;
    } else if (input_X->ndim == 2) {
        input_N = input_X->shape[0];
        input_C = input_X->shape[1];
        input_H = 1;
        input_W = 1;
    }
    const float *input_data = (const float *)(input_X->data);

    // input descriptor
    cudnnTensorDescriptor_t input_desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT, input_N, input_C,
                                          input_H, input_W));

    // output
    float *output_data = (float *)(output_Y->data);

    // output descriptor
    cudnnTensorDescriptor_t output_desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT, input_N, input_C,
                                          input_H, input_W));

    // bn parameter descriptor
    cudnnTensorDescriptor_t bnScaleBiasMeanVar_desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&bnScaleBiasMeanVar_desc));
    CUDNN_CALL(
        cudnnDeriveBNTensorDescriptor(bnScaleBiasMeanVar_desc, input_desc,
                                      CUDNN_BATCHNORM_SPATIAL)); // after conv

    // bn parameter
    const float *bn_scale_data = (const float *)(bn_scale->data);
    const float *bn_bias_data = (const float *)(bn_bias->data);

    const float *estimated_mean_arr = (const float *)estimated_mean->data;
    const float *estimated_var_arr = (const float *)estimated_var->data;

    float alpha = 1.0f;
    float beta = 0.0f;

    CUDNN_CALL(cudnnBatchNormalizationForwardInference(
        cudnn_map[dev_id], CUDNN_BATCHNORM_SPATIAL, &alpha, &beta, input_desc,
        input_data, output_desc, output_data, bnScaleBiasMeanVar_desc,
        bn_scale_data, bn_bias_data, estimated_mean_arr, estimated_var_arr,
        eps));

    CUDNN_CALL(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(bnScaleBiasMeanVar_desc));

    return 0;
}