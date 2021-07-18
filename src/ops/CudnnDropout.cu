#include "gpu_runtime.h"

int CuDNN_DLGpuDropout(const DLArrayHandle input_X, const float dropout,
                       DLArrayHandle output_Y, int *reserve_size,
                       void **reserve_space, int need_allocate,
                       DLStreamHandle stream_handle = NULL) {
    int dev_id = (input_X->ctx).device_id;
    cudnn_init(dev_id, stream_handle);
    size_t input_N, input_C, input_H, input_W;
    // input
    if (input_X->ndim == 2) {
        input_N = input_X->shape[0];
        input_C = input_H = 1;
        input_W = input_X->shape[1];
    } else if (input_X->ndim == 3) {
        input_N = input_X->shape[0];
        input_C = input_X->shape[1];
        input_H = 1;
        input_W = input_X->shape[2];
    } else {
        input_N = input_X->shape[0];
        input_C = input_X->shape[1];
        input_H = input_X->shape[2];
        input_W = input_X->shape[3];
    }
    const float *input_data = (const float *)(input_X->data);
    // input descriptor
    cudnnTensorDescriptor_t input_desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT, input_N, input_C,
                                          input_H, input_W));

    // dropout descriptor
    cudnnDropoutDescriptor_t dropout_desc;
    CUDNN_CALL(cudnnCreateDropoutDescriptor(&dropout_desc));

    unsigned long long seed = 19260817ull;
    size_t state_size;
    CUDNN_CALL(cudnnDropoutGetStatesSize(cudnn_map[dev_id], &state_size));
    if (is_chunk_init(dev_id) == false) {
        chunk_init(dev_id);
    }
    void *state_data = find_chunk(state_size, dev_id);

    CUDNN_CALL(cudnnSetDropoutDescriptor(dropout_desc, cudnn_map[dev_id],
                                         dropout, state_data, state_size,
                                         seed));
    // output
    float *output_data = (float *)output_Y->data;
    // output descriptor
    cudnnTensorDescriptor_t output_desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT, input_N, input_C,
                                          input_H, input_W));

    if (need_allocate == 2)
        del_chunk(*reserve_space, dev_id);

    if (need_allocate > 0) {
        CUDNN_CALL(cudnnDropoutGetReserveSpaceSize(input_desc,
                                                   (size_t *)reserve_size));
        *reserve_space = find_chunk(*reserve_size, dev_id);
    }

    // dropout_forward
    CUDNN_CALL(cudnnDropoutForward(cudnn_map[dev_id], dropout_desc, input_desc,
                                   input_data, output_desc, output_data,
                                   *reserve_space, *reserve_size));

    del_chunk(state_data, dev_id);
    CUDNN_CALL(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_CALL(cudnnDestroyDropoutDescriptor(dropout_desc));
    return 0;
}

int CuDNN_DLGpuDropout_gradient(const DLArrayHandle output_Y,
                                const float dropout, DLArrayHandle input_X,
                                int *reserve_size, void **reserve_space,
                                DLStreamHandle stream_handle = NULL) {
    int dev_id = (input_X->ctx).device_id;
    cudnn_init(dev_id, stream_handle);
    size_t input_N, input_C, input_H, input_W;
    // input
    if (input_X->ndim == 2) {
        input_N = input_X->shape[0];
        input_C = input_H = 1;
        input_W = input_X->shape[1];
    } else if (input_X->ndim == 3) {
        input_N = input_X->shape[0];
        input_C = input_X->shape[1];
        input_H = 1;
        input_W = input_X->shape[2];
    } else {
        input_N = input_X->shape[0];
        input_C = input_X->shape[1];
        input_H = input_X->shape[2];
        input_W = input_X->shape[3];
    }
    float *input_data = (float *)(input_X->data);

    // input descriptor
    cudnnTensorDescriptor_t input_desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT, input_N, input_C,
                                          input_H, input_W));

    // dropout descriptor
    cudnnDropoutDescriptor_t dropout_desc;
    CUDNN_CALL(cudnnCreateDropoutDescriptor(&dropout_desc));

    unsigned long long seed = 19260817ull; // ha
    size_t state_size;

    CUDNN_CALL(cudnnDropoutGetStatesSize(cudnn_map[dev_id], &state_size));

    if (is_chunk_init(dev_id) == false) {
        chunk_init(dev_id);
    }
    void *state_data = find_chunk(state_size, dev_id);
    CUDNN_CALL(cudnnSetDropoutDescriptor(dropout_desc, cudnn_map[dev_id],
                                         dropout, state_data, state_size,
                                         seed));

    // output
    const float *output_data = (const float *)(output_Y->data);

    // output descriptor
    cudnnTensorDescriptor_t output_desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT, input_N, input_C,
                                          input_H, input_W));

    // dropout_backward
    CUDNN_CALL(cudnnDropoutBackward(cudnn_map[dev_id], dropout_desc,
                                    output_desc, output_data, input_desc,
                                    input_data, *reserve_space, *reserve_size));

    del_chunk(state_data, dev_id);
    CUDNN_CALL(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_CALL(cudnnDestroyDropoutDescriptor(dropout_desc));

    return 0;
}
