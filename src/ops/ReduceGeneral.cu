#include "gpu_runtime.h"

int DLGpuReduceGeneral(cudnnReduceTensorOp_t red_type,
                       const DLArrayHandle in_arr, DLArrayHandle out_arr,
                       int *axes, int num_ax,
                       DLStreamHandle stream_handle = NULL, int offset = 0) {
    int dev_id = (in_arr->ctx).device_id;
    cudnn_init(dev_id, stream_handle);

    float one = 1.0f;
    float zero = 0.0f;
    cudnnReduceTensorDescriptor_t rtd;
    CUDNN_CALL(cudnnCreateReduceTensorDescriptor(&rtd));
    CUDNN_CALL(cudnnSetReduceTensorDescriptor(
        rtd, red_type, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN,
        CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES));

    cudnnTensorDescriptor_t adesc;
    cudnnTensorDescriptor_t cdesc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&adesc));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&cdesc));

    int ori_ndim = in_arr->ndim;
    int ndim = max(ori_ndim, 4);
    size_t cpu_mem = ndim * sizeof(int);
    int *dimA = (int *)malloc(cpu_mem);
    int *strideA = (int *)malloc(cpu_mem);
    int *dimC = (int *)malloc(cpu_mem);
    int *strideC = (int *)malloc(cpu_mem);

    for (int i = 0; i < ori_ndim; ++i) {
        dimA[i] = dimC[i] = (int)in_arr->shape[i];
    }
    for (int i = ori_ndim; i < ndim; ++i) {
        dimA[i] = dimC[i] = 1;
    }
    for (int i = 0; i < num_ax; ++i) {
        assert(axes[i] < ori_ndim && axes[i] >= 0);
        dimC[axes[i]] = 1;
    }
    int temp_strideA = 1;
    int temp_strideC = 1;
    for (int i = ndim - 1; i >= 0; --i) {
        strideA[i] = temp_strideA;
        strideC[i] = temp_strideC;
        temp_strideA *= dimA[i];
        temp_strideC *= dimC[i];
    }

    size_t size = temp_strideA * sizeof(float);

    if (is_chunk_init(dev_id) == false) {
        chunk_init(dev_id);
    }
    void *workspace = find_chunk(size, dev_id);

    CUDNN_CALL(cudnnSetTensorNdDescriptor(adesc, CUDNN_DATA_FLOAT, ndim, dimA,
                                          strideA));
    CUDNN_CALL(cudnnSetTensorNdDescriptor(cdesc, CUDNN_DATA_FLOAT, ndim, dimC,
                                          strideC));
    void *out_ptr = (void *)out_arr->data;
    if (offset > 0) {
        size_t fsize = offset * sizeof(float);
        out_ptr = out_ptr + fsize;
    }
    CUDNN_CALL(cudnnReduceTensor(cudnn_map[dev_id], rtd, NULL, 0, workspace,
                                 size, &one, adesc, (const void *)in_arr->data,
                                 &zero, cdesc, out_ptr));

    del_chunk(workspace, dev_id);
    CUDNN_CALL(cudnnDestroyTensorDescriptor(adesc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(cdesc));
    CUDNN_CALL(cudnnDestroyReduceTensorDescriptor(rtd));
    free(dimA);
    free(dimC);
    free(strideA);
    free(strideC);
    return 0;
}

int DLGpuReduceMean(const DLArrayHandle in_arr, DLArrayHandle out_arr,
                    int *axes, int num_ax,
                    DLStreamHandle stream_handle = NULL) {
    return DLGpuReduceGeneral(CUDNN_REDUCE_TENSOR_AVG, in_arr, out_arr, axes,
                              num_ax, stream_handle);
}

// int DLGpuReduceSum(const DLArrayHandle in_arr, DLArrayHandle out_arr, int *axes,
//                    int num_ax, DLStreamHandle stream_handle = NULL) {
//     return DLGpuReduceGeneral(CUDNN_REDUCE_TENSOR_ADD, in_arr, out_arr, axes,
//                               num_ax, stream_handle);
// }

int DLGpuReduceMul(const DLArrayHandle in_arr, DLArrayHandle out_arr, int *axes,
                   int num_ax, DLStreamHandle stream_handle = NULL) {
    return DLGpuReduceGeneral(CUDNN_REDUCE_TENSOR_MUL, in_arr, out_arr, axes,
                              num_ax, stream_handle);
}

int DLGpuReduceMin(const DLArrayHandle in_arr, DLArrayHandle out_arr, int *axes,
                   int num_ax, DLStreamHandle stream_handle = NULL) {
    return DLGpuReduceGeneral(CUDNN_REDUCE_TENSOR_MIN, in_arr, out_arr, axes,
                              num_ax, stream_handle);
}

int DLGpuReduceNorm1(const DLArrayHandle in_arr, DLArrayHandle out_arr,
                     int *axes, int num_ax,
                     DLStreamHandle stream_handle = NULL) {
    return DLGpuReduceGeneral(CUDNN_REDUCE_TENSOR_NORM1, in_arr, out_arr, axes,
                              num_ax, stream_handle);
}

int DLGpuReduceNorm2(const DLArrayHandle in_arr, DLArrayHandle out_arr,
                     int *axes, int num_ax,
                     DLStreamHandle stream_handle = NULL) {
    return DLGpuReduceGeneral(CUDNN_REDUCE_TENSOR_NORM2, in_arr, out_arr, axes,
                              num_ax, stream_handle);
}

int DLGpuReduceNorm2Raw(const DLArrayHandle in_arr, DLArrayHandle out_arr,
                        int *axes, int num_ax, int offset = 0,
                        DLStreamHandle stream_handle = NULL) {
    if (in_arr->ndim == 1 && in_arr->shape[0] == 1) {
        size_t fsize = sizeof(float);
        void *out_ptr = (void *)out_arr->data;
        if (offset > 0) {
            size_t foffset = fsize * offset;
            out_ptr = out_ptr + foffset;
        }
        cudaStream_t cu_stream = static_cast<cudaStream_t>(
            stream_handle ? *(cudaStream_t *)(stream_handle->handle) : NULL);
        cudaMemcpyAsync((void *)in_arr->data, out_ptr, fsize,
                        cudaMemcpyDeviceToDevice, cu_stream);
        return 0;
    } else {
        return DLGpuReduceGeneral(CUDNN_REDUCE_TENSOR_NORM2, in_arr, out_arr,
                                  axes, num_ax, stream_handle, offset);
    }
}
