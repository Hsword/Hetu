#include "gpu_runtime.h"

__global__ void binary_cross_entropy_with_logits_kernel(const float *prediction,
                                                        const float *label,
                                                        float *loss,
                                                        size_t size) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= size)
        return;
    float cur_pred = prediction[id];
    float cur_label = label[id];
    assert(cur_label >= 0 && cur_label <= 1);
    float max_val = max(0., -cur_pred);
    loss[id] = (1 - cur_label) * cur_pred + max_val
               + log(exp(-max_val) + exp(-cur_pred - max_val));
}

int DLGpuBinaryCrossEntropyWithLogits(const DLArrayHandle prediction,
                                      const DLArrayHandle label,
                                      DLArrayHandle loss,
                                      DLStreamHandle stream_handle = NULL) {
    size_t indim = prediction->ndim;
    assert(indim == label->ndim && indim == loss->ndim);
    size_t size = ArrSize(prediction);

    const float *prediction_data = (const float *)prediction->data;
    const float *label_data = (const float *)label->data;
    float *output_data = (float *)loss->data;

    dim3 blocks;
    dim3 threads;
    ThreadBlock1D(threads, blocks, size);
    if (stream_handle) {
        binary_cross_entropy_with_logits_kernel<<<
            blocks, threads, 0, *(cudaStream_t *)stream_handle->handle>>>(
            prediction_data, label_data, output_data, size);
    } else {
        binary_cross_entropy_with_logits_kernel<<<blocks, threads>>>(
            prediction_data, label_data, output_data, size);
    }
    return 0;
}

__global__ void binary_cross_entropy_with_logits_gradient_kernel(
    const float *prediction, const float *label, const float *output_grad,
    float *output, size_t size) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= size)
        return;
    output[id] = output_grad[id] * (1 / (1 + exp(-prediction[id])) - label[id]);
}

int DLGpuBinaryCrossEntropyWithLogits_Gradient(
    const DLArrayHandle prediction, const DLArrayHandle label,
    const DLArrayHandle output_grad, DLArrayHandle output,
    DLStreamHandle stream_handle = NULL) {
    size_t indim = prediction->ndim;
    assert(indim == label->ndim && indim == output_grad->ndim
           && indim == output->ndim);
    size_t size = ArrSize(prediction);

    const float *prediction_data = (const float *)prediction->data;
    const float *label_data = (const float *)label->data;
    const float *output_grad_data = (const float *)output_grad->data;
    float *output_data = (float *)output->data;

    dim3 blocks;
    dim3 threads;
    ThreadBlock1D(threads, blocks, size);
    if (stream_handle) {
        binary_cross_entropy_with_logits_gradient_kernel<<<
            blocks, threads, 0, *(cudaStream_t *)stream_handle->handle>>>(
            prediction_data, label_data, output_grad_data, output_data, size);
    } else {
        binary_cross_entropy_with_logits_gradient_kernel<<<blocks, threads>>>(
            prediction_data, label_data, output_grad_data, output_data, size);
    }
    return 0;
}
