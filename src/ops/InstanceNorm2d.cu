#include "gpu_runtime.h"

__global__ void minus_mean_n_square_kernel1(const float *in_arr,
                                            const float *mean, float *out_arr,
                                            int last_2dim, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    float temp = in_arr[ind] - mean[ind / last_2dim];
    out_arr[ind] = temp * temp;
}

__global__ void std_normal_transform(const float *in_arr, const float *mean_arr,
                                     const float *var_arr, float *out_arr,
                                     int last_2dim, float eps, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    size_t mo_ind = ind / last_2dim;
    out_arr[ind] =
        (in_arr[ind] - mean_arr[mo_ind]) / sqrtf(var_arr[mo_ind] + eps);
}

int DLGpuInstanceNormalization2d(const DLArrayHandle in_arr,
                                 DLArrayHandle mean_arr, DLArrayHandle var_arr,
                                 DLArrayHandle out_arr, float eps,
                                 DLStreamHandle stream_handle) {
    int dev_id = (in_arr->ctx).device_id;
    cudaSetDevice(dev_id);
    cudnn_init(dev_id, stream_handle);

    float one = 1.0f;
    float zero = 0.0f;

    cudnnReduceTensorDescriptor_t rtd;
    CUDNN_CALL(cudnnCreateReduceTensorDescriptor(&rtd));
    CUDNN_CALL(cudnnSetReduceTensorDescriptor(
        rtd, CUDNN_REDUCE_TENSOR_AVG, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN,
        CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES));

    cudnnTensorDescriptor_t adesc;
    cudnnTensorDescriptor_t cdesc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&adesc));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&cdesc));

    int ndim = in_arr->ndim;
    assert(ndim == 4);
    int last_2dim = in_arr->shape[ndim - 1] * in_arr->shape[ndim - 2];
    size_t cpu_mem = ndim * sizeof(int);
    int *dimA = (int *)malloc(cpu_mem);
    int *strideA = (int *)malloc(cpu_mem);
    int *dimC = (int *)malloc(cpu_mem);
    int *strideC = (int *)malloc(cpu_mem);

    int temp_strideA = 1;
    int temp_strideC = 1;

    for (int i = ndim - 1; i >= 0; --i) {
        dimA[i] = (int)in_arr->shape[i];
        dimC[i] = i < in_arr->ndim - 2 ? (int)in_arr->shape[i] : 1;
        strideA[i] = temp_strideA;
        strideC[i] = temp_strideC;
        temp_strideA *= dimA[i];
        temp_strideC *= dimC[i];
    }
    size_t size = temp_strideA * sizeof(float);

    CUDNN_CALL(cudnnSetTensorNdDescriptor(adesc, CUDNN_DATA_FLOAT, ndim, dimA,
                                          strideA));
    CUDNN_CALL(cudnnSetTensorNdDescriptor(cdesc, CUDNN_DATA_FLOAT, ndim, dimC,
                                          strideC));

    CUDNN_CALL(cudnnReduceTensor(cudnn_map[dev_id], rtd, NULL, 0,
                                 (void *)out_arr->data, size, &one, adesc,
                                 (const void *)in_arr->data, &zero, cdesc,
                                 (void *)mean_arr->data));

    dim3 blocks;
    dim3 threads;
    if (temp_strideA <= 1024) {
        threads.x = temp_strideA;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (temp_strideA + 1023) / 1024;
    }

    if (stream_handle)
        minus_mean_n_square_kernel1<<<blocks, threads, 0,
                                      *(cudaStream_t *)stream_handle->handle>>>(
            (const float *)in_arr->data, (const float *)mean_arr->data,
            (float *)out_arr->data, last_2dim, temp_strideA);
    else
        minus_mean_n_square_kernel1<<<blocks, threads>>>(
            (const float *)in_arr->data, (const float *)mean_arr->data,
            (float *)out_arr->data, last_2dim, temp_strideA);

    CUDNN_CALL(cudnnReduceTensor(cudnn_map[dev_id], rtd, NULL, 0,
                                 (void *)out_arr->data, size, &one, adesc,
                                 (const void *)out_arr->data, &zero, cdesc,
                                 (void *)var_arr->data));

    if (temp_strideA <= 1024) {
        threads.x = temp_strideA;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (temp_strideA + 1023) / 1024;
    }
    if (stream_handle)
        std_normal_transform<<<blocks, threads, 0,
                               *(cudaStream_t *)stream_handle->handle>>>(
            (const float *)in_arr->data, (const float *)mean_arr->data,
            (const float *)var_arr->data, (float *)out_arr->data, last_2dim,
            eps, temp_strideA);
    else
        std_normal_transform<<<blocks, threads>>>(
            (const float *)in_arr->data, (const float *)mean_arr->data,
            (const float *)var_arr->data, (float *)out_arr->data, last_2dim,
            eps, temp_strideA);

    CUDNN_CALL(cudnnDestroyTensorDescriptor(adesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(cdesc));
    CUDNN_CALL(cudnnDestroyReduceTensorDescriptor(rtd));
    free(dimA);
    free(dimC);
    free(strideA);
    free(strideC);
    return 0;
}

__global__ void calculate_grad_kernel(const float *out_grads,
                                      const float *in_arr,
                                      const float *mean_arr,
                                      const float *var_arr, float *grad_arr,
                                      size_t last2dim, float eps, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    size_t mo_ind = ind / last2dim;
    float y = (in_arr[ind] - mean_arr[mo_ind]) / sqrtf(var_arr[mo_ind] + eps);
    grad_arr[ind] = out_grads[ind] * (1.0 - 1.0 / (float)last2dim - y * y)
                    / sqrtf(var_arr[mo_ind] + eps);
}

int DLGpuInstanceNormalization2dGradient(const DLArrayHandle out_grads,
                                         const DLArrayHandle in_arr,
                                         DLArrayHandle grad_arr,
                                         const DLArrayHandle mean_arr,
                                         const DLArrayHandle var_arr, float eps,
                                         DLStreamHandle stream_handle) {
    /*
       already have mean and var, we directly get y = x-u / sigma
       the grad_arr = out_grad * (1 - 1/WH - y^2) / sigma
     */
    int dev_id = (out_grads->ctx).device_id;
    cudnn_init(dev_id, stream_handle);

    int ndim = out_grads->ndim;
    assert(ndim == 4);
    int total_elements = 1;
    for (int i = 0; i < ndim; ++i)
        total_elements *= out_grads->shape[i];
    int last2dim = out_grads->shape[ndim - 1] * out_grads->shape[ndim - 2];

    dim3 blocks;
    dim3 threads;
    if (total_elements <= 1024) {
        threads.x = total_elements;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (total_elements + 1023) / 1024;
    }

    if (stream_handle)
        calculate_grad_kernel<<<blocks, threads, 0,
                                *(cudaStream_t *)stream_handle->handle>>>(
            (const float *)out_grads->data, (const float *)in_arr->data,
            (const float *)mean_arr->data, (const float *)var_arr->data,
            (float *)grad_arr->data, last2dim, eps, total_elements);
    else
        calculate_grad_kernel<<<blocks, threads>>>(
            (const float *)out_grads->data, (const float *)in_arr->data,
            (const float *)mean_arr->data, (const float *)var_arr->data,
            (float *)grad_arr->data, last2dim, eps, total_elements);

    return 0;
}
