#include "gpu_runtime.h"

__global__ void init_mean_and_var_kernel(float *mean, float *var, size_t size) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= size)
        return;
    mean[id] = 0;
    var[id] = 0;
}

__global__ void copy_mean_and_var_kernel(float *mean, float *var,
                                         float *saved_mean, float *saved_var,
                                         size_t size) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= size)
        return;
    saved_mean[id] = mean[id];
    saved_var[id] = var[id];
}

// the shape of bn_scale/bias   1*C*1*1
int CuDNN_DLGpuBatch_Normalization(
    const DLArrayHandle input_X, const DLArrayHandle bn_scale,
    const DLArrayHandle bn_bias, DLArrayHandle output_Y, float momentum,
    float eps, DLArrayHandle save_mean_arr, DLArrayHandle save_var_arr,
    DLArrayHandle running_mean_arr, DLArrayHandle running_var_arr,
    DLStreamHandle stream_handle = NULL) {
    int dev_id = (input_X->ctx).device_id;
    cudaSetDevice(dev_id);
    cudnn_init(dev_id, stream_handle);

    // input
    size_t input_N = input_X->shape[0];
    size_t input_C = input_X->shape[1];
    size_t input_H = input_X->shape[2];
    size_t input_W = input_X->shape[3];
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

    void *save_mean = save_mean_arr->data;
    void *save_var = save_var_arr->data;
    void *running_mean = running_mean_arr->data;
    void *running_var = running_var_arr->data;
    float alpha = 1.0f;
    float beta = 0.0f;

    CUDNN_CALL(cudnnBatchNormalizationForwardTraining(
        cudnn_map[dev_id], CUDNN_BATCHNORM_SPATIAL, &alpha, &beta, input_desc,
        input_data, output_desc, output_data, bnScaleBiasMeanVar_desc,
        bn_scale_data, bn_bias_data, momentum, (float *)save_mean,
        (float *)save_var, eps, running_mean, running_var));

    CUDNN_CALL(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(bnScaleBiasMeanVar_desc));

    return 0;
}

int CuDNN_DLGpuBatch_Normalization_gradient(
    const DLArrayHandle gradient_Y, const DLArrayHandle input_X,
    const DLArrayHandle bn_scale, DLArrayHandle gradient_X,
    DLArrayHandle gradient_bn_scale, DLArrayHandle gradient_bn_bias, float eps,
    DLArrayHandle running_mean_arr, DLArrayHandle running_var_arr,
    DLStreamHandle stream_handle = NULL) {
    int dev_id = (input_X->ctx).device_id;
    cudnn_init(dev_id, stream_handle);

    // input
    size_t input_N = input_X->shape[0];
    size_t input_C = input_X->shape[1];
    size_t input_H = input_X->shape[2];
    size_t input_W = input_X->shape[3];
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
    void *running_mean = (float *)(running_mean_arr->data);
    void *running_var = (float *)(running_var_arr->data);
    float one = 1.0f;
    float zero = 0.0f;

    CUDNN_CALL(cudnnBatchNormalizationBackward(
        cudnn_map[dev_id], CUDNN_BATCHNORM_SPATIAL_PERSISTENT, &one, &zero,
        &one, &zero, input_desc, input_data, output_desc, gradient_y_data,
        input_desc, gradient_x_data, bnScaleBiasMeanVar_desc, bn_scale_data,
        gradient_bn_scale_data, gradient_bn_bias_data, eps, running_mean,
        running_var));

    CUDNN_CALL(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(bnScaleBiasMeanVar_desc));
    return 0;
}

int CuDNN_DLGpuBatch_Normalization_inference(
    const DLArrayHandle input_X, const DLArrayHandle bn_scale,
    const DLArrayHandle bn_bias, DLArrayHandle output_Y, float eps,
    DLArrayHandle save_mean_arr, DLArrayHandle save_var_arr,
    DLStreamHandle stream_handle = NULL) {
    int dev_id = (input_X->ctx).device_id;
    cudaSetDevice(dev_id);
    cudnn_init(dev_id, stream_handle);

    // input
    size_t input_N = input_X->shape[0];
    size_t input_C = input_X->shape[1];
    size_t input_H = input_X->shape[2];
    size_t input_W = input_X->shape[3];
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

    const float *save_mean = (const float *)save_mean_arr->data;
    const float *save_var = (const float *)save_var_arr->data;

    float alpha = 1.0f;
    float beta = 0.0f;

    CUDNN_CALL(cudnnBatchNormalizationForwardInference(
        cudnn_map[dev_id], CUDNN_BATCHNORM_SPATIAL, &alpha, &beta, input_desc,
        input_data, output_desc, output_data, bnScaleBiasMeanVar_desc,
        bn_scale_data, bn_bias_data, save_mean, save_var, eps));

    CUDNN_CALL(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(bnScaleBiasMeanVar_desc));

    return 0;
}