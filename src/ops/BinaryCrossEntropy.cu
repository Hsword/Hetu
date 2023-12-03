#include "gpu_runtime.h"

const float EPSILON = 1e-12;

// -label * log(prediction) - (1 - label) * log(1 - prediction)
__global__ void binary_cross_entropy_kernel(int nrow, const float *prediction,
                                            const float *label, float *loss) {
    // Two dimensional thread blocks.
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= nrow)
        return;
    float cur_pred = prediction[id];
    float cur_label = label[id];
    assert(cur_label >= 0 && cur_label <= 1);
    float log_input_val = log(cur_pred + EPSILON);
    float log_1_minus_input_val = log(1 - cur_pred + EPSILON);
    // float log_input_val = log(cur_pred);
    // float log_1_minus_input_val = log(1 - cur_pred);

    // float neg_100 = -100;
    // log_input_val = max(log_input_val, neg_100);
    // log_1_minus_input_val = max(log_1_minus_input_val, neg_100);
    loss[id] =
        ((cur_label - 1) * log_1_minus_input_val) - (cur_label * log_input_val);
}

int DLGpuBinaryCrossEntropy(const DLArrayHandle prediction,
                            const DLArrayHandle label, DLArrayHandle loss,
                            DLStreamHandle stream_handle = NULL) {
    size_t indim = prediction->ndim;
    assert(indim == label->ndim && indim == loss->ndim);
    int nrow = 1;
    for (int i = 0; i < indim - 1; ++i) {
        nrow *= prediction->shape[i];
    }

    const float *prediction_data = (const float *)prediction->data;
    const float *label_data = (const float *)label->data;
    float *output_data = (float *)loss->data;

    dim3 blocks;
    dim3 threads;
    if (nrow <= 1024) {
        threads.x = nrow;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (nrow + 1023) / 1024;
    }
    // 1 block
    if (stream_handle) {
        binary_cross_entropy_kernel<<<blocks, threads, 0,
                                      *(cudaStream_t *)stream_handle->handle>>>(
            nrow, prediction_data, label_data, output_data);
    } else {
        binary_cross_entropy_kernel<<<blocks, threads>>>(
            nrow, prediction_data, label_data, output_data);
    }
    return 0;
}

__global__ void binary_cross_entropy_gradient_kernel(int nrow,
                                                     const float *prediction,
                                                     const float *label,
                                                     const float *output_grad,
                                                     float *output) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= nrow)
        return;
    float cur_pred = prediction[id];
    output[id] = output_grad[id] * (cur_pred - label[id])
                 / max((1 - cur_pred) * cur_pred, EPSILON);
}

int DLGpuBinaryCrossEntropy_Gradient(const DLArrayHandle prediction,
                                     const DLArrayHandle label,
                                     const DLArrayHandle output_grad,
                                     DLArrayHandle output,
                                     DLStreamHandle stream_handle = NULL) {
    size_t indim = prediction->ndim;
    assert(indim >= 2 && indim == label->ndim && indim == output_grad->ndim
           && indim == output->ndim);
    int nrow = 1;
    for (int i = 0; i < indim - 1; ++i) {
        nrow *= prediction->shape[i];
    }

    const float *prediction_data = (const float *)prediction->data;
    const float *label_data = (const float *)label->data;
    const float *output_grad_data = (const float *)output_grad->data;
    float *output_data = (float *)output->data;

    dim3 blocks;
    dim3 threads;
    if (nrow <= 1024) {
        threads.x = nrow;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (nrow + 1023) / 1024;
    }
    if (stream_handle) {
        binary_cross_entropy_gradient_kernel<<<
            blocks, threads, 0, *(cudaStream_t *)stream_handle->handle>>>(
            nrow, prediction_data, label_data, output_grad_data, output_data);
    } else {
        binary_cross_entropy_gradient_kernel<<<blocks, threads>>>(
            nrow, prediction_data, label_data, output_grad_data, output_data);
    }
    return 0;
}
