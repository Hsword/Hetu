#include "gpu_runtime.h"

int CuDNN_DLGpuConv2d(const DLArrayHandle input_x, const DLArrayHandle input_f,
                      DLArrayHandle output, const int padding_h,
                      const int padding_w, const int stride_h,
                      const int stride_w, DLStreamHandle stream_handle = NULL) {
    int dev_id = (input_x->ctx).device_id;
    cudnn_init(dev_id, stream_handle);
    size_t input_N = input_x->shape[0];
    size_t input_C = input_x->shape[1];
    size_t input_H = input_x->shape[2];
    size_t input_W = input_x->shape[3];
    const float *input_data = (const float *)input_x->data;

    // input
    cudnnTensorDescriptor_t input_desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT, input_N, input_C,
                                          input_H, input_W));
    size_t filter_N = input_f->shape[0];
    size_t filter_C = input_f->shape[1];
    size_t filter_H = input_f->shape[2];
    size_t filter_W = input_f->shape[3];
    const float *filter_data = (const float *)input_f->data;

    // filter
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT,
                                          CUDNN_TENSOR_NCHW, filter_N, filter_C,
                                          filter_H, filter_W));

    // convolution
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_CALL(cudnnSetConvolution2dDescriptor(
        conv_desc, padding_h, padding_w, stride_h, stride_w, 1, 1,
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    size_t out_N = output->shape[0];
    size_t out_C = output->shape[1];
    size_t out_H = output->shape[2];
    size_t out_W = output->shape[3];
    // output
    cudnnTensorDescriptor_t out_desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT, out_N, out_C, out_H,
                                          out_W));
    float *output_data = (float *)output->data;

    // search for the best algorithm
    int request_cnt = 9, return_cnt = 9;
    cudnnConvolutionFwdAlgoPerf_t algo_perf[9];
    CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm_v7(
        cudnn_map[dev_id], input_desc, filter_desc, conv_desc, out_desc,
        request_cnt, &return_cnt, algo_perf));

    if (is_chunk_init(dev_id) == false)
        chunk_init(dev_id);

    size_t workspace_size;
    void *work_data = nullptr;
    cudnnConvolutionFwdAlgo_t algo;
    for(int i = 0; i < return_cnt; ++i) {
        CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(
            cudnn_map[dev_id], input_desc, filter_desc, conv_desc, out_desc, algo_perf[i].algo,
            &workspace_size));
        work_data = find_chunk(workspace_size, dev_id, false);
        if (work_data) {
            algo = algo_perf[i].algo;
            break;
        }
    }

    float alpha = 1.0f;
    float beta = 0.0f;
    CUDNN_CALL(cudnnConvolutionForward(
        cudnn_map[dev_id], &alpha, input_desc, input_data, filter_desc,
        filter_data, conv_desc, algo, work_data, workspace_size, &beta,
        out_desc, output_data));
    del_chunk(work_data, dev_id);
    CUDNN_CALL(cudnnDestroyTensorDescriptor(out_desc));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
    CUDNN_CALL(cudnnDestroyFilterDescriptor(filter_desc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(input_desc));
    return 0;
}
int CuDNN_DLGpuConv2d_Gradient_of_Filter(const DLArrayHandle input_x,
                                         const DLArrayHandle gradient_y,
                                         DLArrayHandle gradient_f,
                                         const int padding_h,
                                         const int padding_w,
                                         const int stride_h, const int stride_w,
                                         DLStreamHandle stream_handle = NULL) {
    // create handle
    int dev_id = (input_x->ctx).device_id;
    cudnn_init(dev_id, stream_handle);

    // input
    size_t input_N = input_x->shape[0];
    size_t input_C = input_x->shape[1];
    size_t input_H = input_x->shape[2];
    size_t input_W = input_x->shape[3];
    const float *input_data = (const float *)input_x->data;

    cudnnTensorDescriptor_t input_desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT, input_N, input_C,
                                          input_H, input_W));
    // dy
    size_t dy_N = gradient_y->shape[0];
    size_t dy_C = gradient_y->shape[1];
    size_t dy_H = gradient_y->shape[2];
    size_t dy_W = gradient_y->shape[3];
    const float *dy_data = (const float *)gradient_y->data;

    cudnnTensorDescriptor_t dy_desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&dy_desc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(
        dy_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, dy_N, dy_C, dy_H, dy_W));

    // conv2d
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_CALL(cudnnSetConvolution2dDescriptor(
        conv_desc, padding_h, padding_w, stride_h, stride_w, 1, 1,
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    // dw
    size_t df_N = gradient_f->shape[0];
    size_t df_C = gradient_f->shape[1];
    size_t df_H = gradient_f->shape[2];
    size_t df_W = gradient_f->shape[3];
    float *df_data = (float *)gradient_f->data;

    cudnnFilterDescriptor_t df_desc;
    CUDNN_CALL(cudnnCreateFilterDescriptor(&df_desc));
    CUDNN_CALL(cudnnSetFilter4dDescriptor(
        df_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, df_N, df_C, df_H, df_W));

    // search for the best algorithm
    int request_cnt = 9, return_cnt = 9;
    cudnnConvolutionBwdFilterAlgoPerf_t algo_perf[9];
    CUDNN_CALL(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
        cudnn_map[dev_id], input_desc, dy_desc, conv_desc, df_desc,
        request_cnt, &return_cnt, algo_perf));

    cudaError_t err;
    size_t workspace_size;
    void *work_data = nullptr;
    cudnnConvolutionBwdFilterAlgo_t algo;
    for(int i = 0; i < return_cnt; ++i) {
        CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
            cudnn_map[dev_id], input_desc, dy_desc, conv_desc, df_desc, algo_perf[i].algo,
            &workspace_size));
        err = cudaMalloc(&work_data, workspace_size);
        if (err == cudaSuccess) {
            algo = algo_perf[i].algo;
            break;
        }
    }

    float alpha = 1.0f;
    float beta = 0.0f;
    CUDNN_CALL(cudnnConvolutionBackwardFilter(
        cudnn_map[dev_id], &alpha, input_desc, input_data, dy_desc, dy_data,
        conv_desc, algo, work_data, workspace_size, &beta, df_desc, df_data));
    cudaFree(work_data);
    CUDNN_CALL(cudnnDestroyTensorDescriptor(dy_desc));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
    CUDNN_CALL(cudnnDestroyFilterDescriptor(df_desc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(input_desc));
    return 0;
}

int CuDNN_DLGpuConv2d_Gradient_of_Data(const DLArrayHandle input_f,
                                       const DLArrayHandle gradient_y,
                                       DLArrayHandle gradient_x,
                                       const int padding_h, const int padding_w,
                                       const int stride_h, const int stride_w,
                                       DLStreamHandle stream_handle = NULL) {
    // create handle
    int dev_id = (input_f->ctx).device_id;
    cudnn_init(dev_id, stream_handle);

    // filter
    size_t filter_N = input_f->shape[0];
    size_t filter_C = input_f->shape[1];
    size_t filter_H = input_f->shape[2];
    size_t filter_W = input_f->shape[3];
    const float *filter_data = (const float *)input_f->data;

    cudnnFilterDescriptor_t filter_desc;
    CUDNN_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT,
                                          CUDNN_TENSOR_NCHW, filter_N, filter_C,
                                          filter_H, filter_W));
    // dy
    size_t dy_N = gradient_y->shape[0];
    size_t dy_C = gradient_y->shape[1];
    size_t dy_H = gradient_y->shape[2];
    size_t dy_W = gradient_y->shape[3];
    const float *dy_data = (const float *)gradient_y->data;

    cudnnTensorDescriptor_t dy_desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&dy_desc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(
        dy_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, dy_N, dy_C, dy_H, dy_W));

    // conv2d
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_CALL(cudnnSetConvolution2dDescriptor(
        conv_desc, padding_h, padding_w, stride_h, stride_w, 1, 1,
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    // dx
    size_t dx_N = gradient_x->shape[0];
    size_t dx_C = gradient_x->shape[1];
    size_t dx_H = gradient_x->shape[2];
    size_t dx_W = gradient_x->shape[3];
    float *dx_data = (float *)gradient_x->data;

    cudnnTensorDescriptor_t dx_desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&dx_desc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(
        dx_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, dx_N, dx_C, dx_H, dx_W));

    // search for the best algorithm
    int request_cnt = 9, return_cnt = 9;
    cudnnConvolutionBwdDataAlgoPerf_t algo_perf[9];
    CUDNN_CALL(cudnnGetConvolutionBackwardDataAlgorithm_v7(
        cudnn_map[dev_id], filter_desc, dy_desc, conv_desc, dx_desc,
        request_cnt, &return_cnt, algo_perf));

    cudaError_t err;
    size_t workspace_size;
    void *work_data = nullptr;
    cudnnConvolutionBwdDataAlgo_t algo;
    for(int i = 0; i < return_cnt; ++i) {
        CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
            cudnn_map[dev_id], filter_desc, dy_desc, conv_desc, dx_desc, algo_perf[i].algo,
            &workspace_size));
        err = cudaMalloc(&work_data, workspace_size);
        if (err == cudaSuccess) {
            algo = algo_perf[i].algo;
            break;
        }
    }

    float alpha = 1.0f;
    float beta = 0.0f;
    CUDNN_CALL(cudnnConvolutionBackwardData(
        cudnn_map[dev_id], &alpha, filter_desc, filter_data, dy_desc, dy_data,
        conv_desc, algo, work_data, workspace_size, &beta, dx_desc, dx_data));
    cudaFree(work_data);
    CUDNN_CALL(cudnnDestroyTensorDescriptor(dy_desc));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(dx_desc));
    CUDNN_CALL(cudnnDestroyFilterDescriptor(filter_desc));
    return 0;
}
