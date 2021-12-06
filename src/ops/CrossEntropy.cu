#include "gpu_runtime.h"

__global__ void cross_entropy_kernel(const float *y,
                                           const float *y_, float *output,
                                           size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
        return;
    output[idx] = -log(y[idx]) * y_[idx];
}

int DLGpuCrossEntropy(const DLArrayHandle input_y,
                              const DLArrayHandle label, DLArrayHandle output,
                              DLStreamHandle stream_handle = NULL) {
    size_t indim = input_y->ndim;
    assert(indim == label->ndim && indim == output->ndim + 1);
    int dev_id = (input_y->ctx).device_id;
    cudnn_init(dev_id, stream_handle);
    int n_ = 1;
    for (int i = 0; i < indim - 1; ++i) {
        n_ *= input_y->shape[i];
    }
    int c_ = input_y->shape[indim - 1];
    size_t size = n_ * c_;
    const float *y_data = (const float *)(input_y->data);
    float *label_data = (float *)(label->data);
    float *output_data = (float *)(output->data);
    float alpha = 1.0f;
    float beta = 0.0f;
    cudnnTensorDescriptor_t desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&desc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT, n_, c_, 1, 1));
    if (is_chunk_init(dev_id) == false) {
        chunk_init(dev_id);
    }
    void *temp_data = find_chunk(size * sizeof(float), dev_id);

    dim3 blocks;
    dim3 threads;
    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }

    // Compute -log(yi)*yi_
    if (stream_handle) {
        cross_entropy_kernel<<<blocks, threads, 0,
                                     *(cudaStream_t *)stream_handle->handle>>>(
            y_data, label_data, (float *)temp_data, size);
    } else {
        cross_entropy_kernel<<<blocks, threads>>>(
            y_data, label_data, (float *)temp_data, size);
    }

    // Compute Sigma(-log(yi)*yi_)
    cudnnReduceTensorDescriptor_t rtd;
    CUDNN_CALL(cudnnCreateReduceTensorDescriptor(&rtd));
    CUDNN_CALL(cudnnSetReduceTensorDescriptor(
        rtd, CUDNN_REDUCE_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN,
        CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES));
    cudnnTensorDescriptor_t new_desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&new_desc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(new_desc, CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT, n_, 1, 1, 1));
    CUDNN_CALL(cudnnReduceTensor(cudnn_map[dev_id], rtd, NULL, 0, temp_data,
                                 size * sizeof(float), &alpha, desc,
                                 (const void *)temp_data, &beta, new_desc,
                                 (void *)output_data));

    CUDNN_CALL(cudnnDestroyReduceTensorDescriptor(rtd));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(new_desc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(desc));
    del_chunk(temp_data, dev_id);
    return 0;
}

__global__ void cross_entropy_gradient_kernel(const float *y, const float *y_,
                                     const float *grad_data, float *output_data,
                                     int last_dim, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    output_data[ind] = - y_[ind] / y[ind] * grad_data[ind / last_dim];
}

int DLGpuCrossEntropyGradient(const DLArrayHandle grad,
                                      const DLArrayHandle input_y,
                                      const DLArrayHandle label,
                                      DLArrayHandle output,
                                      DLStreamHandle stream_handle = NULL) {
    size_t indim = input_y->ndim;
    assert(indim == label->ndim && indim == output->ndim
           && indim == grad->ndim + 1);

    int n_ = 1;
    for (int i = 0; i < indim - 1; ++i) {
        n_ *= input_y->shape[i];
    }
    int c_ = input_y->shape[indim - 1];
    size_t size = n_ * c_;
    const float *grad_data = (const float *)grad->data;
    const float *y_data = (const float *)input_y->data;
    const float *label_data = (const float *)label->data;
    float *output_data = (float *)output->data;

    dim3 blocks;
    dim3 threads;
    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }
    if (stream_handle) {
        cross_entropy_gradient_kernel<<<
            blocks, threads, 0, *(cudaStream_t *)stream_handle->handle>>>(
            y_data, label_data, grad_data, output_data, c_, size);
    } else {
        cross_entropy_gradient_kernel<<<blocks, threads>>>(
            y_data, label_data, grad_data, output_data, c_, size);
    }
    return 0;
}
