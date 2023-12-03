#include "gpu_runtime.h"

__global__ void less_const_kernel(const float *input, float *output,
                                  float threshold, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    output[ind] = (abs(input[ind]) < threshold);
}

__global__ void less_const_kernel_1d_buffer(const float *input, float *output,
                                            float threshold, size_t dim,
                                            size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    const float *src_ptr = input + ind * dim;
    float result = 0;
    for (size_t i = 0; i < dim; ++i) {
        result += (abs(src_ptr[i]) < threshold);
    }
    output[ind] = result;
}

int DLGpuNumLessThan(const DLArrayHandle input, DLArrayHandle middle,
                     DLArrayHandle output, float threshold, int *axes,
                     int num_ax, DLStreamHandle stream_handle = NULL) {
    size_t size = ArrSize(input);
    size_t dim = input->shape[1];
    const float *input_data = (const float *)input->data;
    float *middle_data = (float *)middle->data;
    if (middle->ndim == 1) {
        dim = size / middle->shape[0];
        size = middle->shape[0];
    }
    dim3 blocks;
    dim3 threads;
    ThreadBlock1D(threads, blocks, size);
    if (middle->ndim == 1) {
        if (stream_handle)
            less_const_kernel_1d_buffer<<<
                blocks, threads, 0, *(cudaStream_t *)stream_handle->handle>>>(
                input_data, middle_data, threshold, dim, size);
        else
            less_const_kernel_1d_buffer<<<blocks, threads>>>(
                input_data, middle_data, threshold, dim, size);
    } else {
        if (stream_handle)
            less_const_kernel<<<blocks, threads, 0,
                                *(cudaStream_t *)stream_handle->handle>>>(
                input_data, middle_data, threshold, size);
        else
            less_const_kernel<<<blocks, threads>>>(input_data, middle_data,
                                                   threshold, size);
    }
    return DLGpuReduceSum(middle, output, axes, num_ax, stream_handle);
}

__global__ void set_less_const_kernel(float *arr, float threshold,
                                      size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    if (abs(arr[ind]) < threshold) {
        arr[ind] = 0;
    }
}

int DLGpuSetLessThan(const DLArrayHandle arr, float threshold,
                     DLStreamHandle stream_handle = NULL) {
    size_t size = ArrSize(arr);
    float *arr_data = (float *)arr->data;
    dim3 blocks;
    dim3 threads;
    ThreadBlock1D(threads, blocks, size);
    if (stream_handle)
        set_less_const_kernel<<<blocks, threads, 0,
                                *(cudaStream_t *)stream_handle->handle>>>(
            arr_data, threshold, size);
    else
        set_less_const_kernel<<<blocks, threads>>>(arr_data, threshold, size);
    return 0;
}

__global__ void set_mask_less_const_kernel(float *arr, int *mask,
                                           float threshold, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    if (abs(arr[ind]) < threshold) {
        arr[ind] = 0.;
        mask[ind] = 0;
    } else {
        mask[ind] = 1;
    }
}

int DLGpuSetMaskLessThan(DLArrayHandle arr, DLArrayHandle mask, float threshold,
                         DLStreamHandle stream_handle = NULL) {
    size_t size = ArrSize(arr);
    float *arr_data = (float *)arr->data;
    int *mask_data = (int *)mask->data;
    dim3 blocks;
    dim3 threads;
    ThreadBlock1D(threads, blocks, size);
    if (stream_handle)
        set_mask_less_const_kernel<<<blocks, threads, 0,
                                     *(cudaStream_t *)stream_handle->handle>>>(
            arr_data, mask_data, threshold, size);
    else
        set_mask_less_const_kernel<<<blocks, threads>>>(arr_data, mask_data,
                                                        threshold, size);
    return 0;
}

__global__ void get_larger_than_kernel_feature_dimension(const float *arr,
                                                         const float *threshold,
                                                         int *mask,
                                                         size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    mask[ind] = (abs(arr[ind]) >= threshold[ind]);
}

__global__ void get_larger_than_kernel_feature(const float *arr,
                                               const float *threshold,
                                               int *mask, size_t size,
                                               size_t dim) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    size_t rind = ind / dim;
    mask[ind] = (abs(arr[ind]) >= threshold[rind]);
}

__global__ void get_larger_than_kernel_dimension(const float *arr,
                                                 const float *threshold,
                                                 int *mask, size_t size,
                                                 size_t dim) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    size_t cind = ind % dim;
    mask[ind] = (abs(arr[ind]) >= threshold[cind]);
}

__global__ void get_larger_than_kernel_global(const float *arr,
                                              const float *threshold, int *mask,
                                              size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    mask[ind] = (abs(arr[ind]) >= threshold[0]);
}

int DLGpuGetLargerThan(const DLArrayHandle input, const DLArrayHandle threshold,
                       DLArrayHandle mask,
                       DLStreamHandle stream_handle = NULL) {
    assert(input->ndim == 2);
    size_t size = ArrSize(input);
    size_t dim = input->shape[1];
    const float *input_data = (const float *)input->data;
    const float *thres_data = (const float *)threshold->data;
    int *mask_data = (int *)mask->data;
    size_t thres_ndim = threshold->ndim;
    size_t last_dim = threshold->shape[thres_ndim - 1];
    bool use_feature = (thres_ndim > 1);
    bool use_dimension = (last_dim > 1);
    dim3 blocks;
    dim3 threads;
    ThreadBlock1D(threads, blocks, size);
    assert(stream_handle != NULL);
    cudaStream_t stream = *(cudaStream_t *)stream_handle->handle;
    if (use_feature && use_dimension) {
        get_larger_than_kernel_feature_dimension<<<blocks, threads, 0,
                                                   stream>>>(
            input_data, thres_data, mask_data, size);
    } else if (!use_feature && use_dimension) {
        get_larger_than_kernel_dimension<<<blocks, threads, 0, stream>>>(
            input_data, thres_data, mask_data, size, dim);
    } else if (use_feature && !use_dimension) {
        get_larger_than_kernel_feature<<<blocks, threads, 0, stream>>>(
            input_data, thres_data, mask_data, size, dim);
    } else {
        get_larger_than_kernel_global<<<blocks, threads, 0, stream>>>(
            input_data, thres_data, mask_data, size);
    }
    return 0;
}

__global__ void less_tensor_kernel_feature_dimension(const float *input,
                                                     float *output,
                                                     const float *threshold,
                                                     size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    output[ind] = (abs(input[ind]) < threshold[ind]);
}

__global__ void less_tensor_kernel_feature(const float *input, float *output,
                                           const float *threshold, size_t size,
                                           size_t dim) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    output[ind] = (abs(input[ind]) < threshold[ind / dim]);
}

__global__ void less_tensor_kernel_dimension(const float *input, float *output,
                                             const float *threshold,
                                             size_t size, size_t dim) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    output[ind] = (abs(input[ind]) < threshold[ind % dim]);
}

__global__ void less_tensor_kernel_global(const float *input, float *output,
                                          const float *threshold, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    output[ind] = (abs(input[ind]) < threshold[0]);
}

int DLGpuNumLessThanTensorThreshold(const DLArrayHandle input,
                                    DLArrayHandle middle, DLArrayHandle output,
                                    const DLArrayHandle threshold, int *axes,
                                    int num_ax,
                                    DLStreamHandle stream_handle = NULL) {
    assert(input->ndim == 2);
    size_t size = ArrSize(input);
    size_t dim = input->shape[1];
    const float *input_data = (const float *)input->data;
    float *middle_data = (float *)middle->data;
    const float *thres_data = (const float *)threshold->data;
    size_t thres_ndim = threshold->ndim;
    size_t last_dim = threshold->shape[thres_ndim - 1];
    bool use_feature = (thres_ndim > 1);
    bool use_dimension = (last_dim > 1);
    dim3 blocks;
    dim3 threads;
    ThreadBlock1D(threads, blocks, size);
    assert(stream_handle != NULL);
    cudaStream_t stream = *(cudaStream_t *)stream_handle->handle;
    if (use_feature && use_dimension) {
        less_tensor_kernel_feature_dimension<<<blocks, threads, 0, stream>>>(
            input_data, middle_data, thres_data, size);
    } else if (!use_feature && use_dimension) {
        less_tensor_kernel_dimension<<<blocks, threads, 0, stream>>>(
            input_data, middle_data, thres_data, size, dim);
    } else if (use_feature && !use_dimension) {
        less_tensor_kernel_feature<<<blocks, threads, 0, stream>>>(
            input_data, middle_data, thres_data, size, dim);
    } else {
        less_tensor_kernel_global<<<blocks, threads, 0, stream>>>(
            input_data, middle_data, thres_data, size);
    }
    return DLGpuReduceSum(middle, output, axes, num_ax, stream_handle);
}

__global__ void multiple_grouping_alpha_kernel(float *arr, const int *grouping,
                                               const float *alpha, size_t size,
                                               size_t dim) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    int group_ind = grouping[ind / dim];
    float cur_alpha = alpha[group_ind * dim + ind % dim];
    arr[ind] *= cur_alpha;
}

int DLGpuMultiplyGroupingAlpha(DLArrayHandle arr, const DLArrayHandle grouping,
                               const DLArrayHandle alpha,
                               DLStreamHandle stream_handle = NULL) {
    assert(arr->ndim == 2);
    size_t size = ArrSize(arr);
    size_t dim = arr->shape[1];
    float *arr_data = (float *)arr->data;
    const int *grouping_data = (const int *)grouping->data;
    const float *alpha_data = (const float *)alpha->data;
    dim3 blocks;
    dim3 threads;
    ThreadBlock1D(threads, blocks, size);
    assert(stream_handle != NULL);
    cudaStream_t stream = *(cudaStream_t *)stream_handle->handle;
    multiple_grouping_alpha_kernel<<<blocks, threads, 0, stream>>>(
        arr_data, grouping_data, alpha_data, size, dim);
    return 0;
}