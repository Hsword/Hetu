#include "gpu_runtime.h"
#include "dispatch.h"

template <typename spec_t>
__global__ void ele_add_kernel(const spec_t *matA, const spec_t *matB,
                               spec_t *output, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    output[ind] = matA[ind] + matB[ind];
}

template <typename spec_t>
__global__ void ele_broadcast_add_kernel(const spec_t *matL, const spec_t *matS,
                                         spec_t *output, size_t size,
                                         uint *L_strides, uint *S_strides,
                                         uint *L_dims, uint *S_dims,
                                         size_t L_ndims, size_t S_ndims) {
    size_t o_ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (o_ind >= size)
        return;

    size_t s_ind = 0;
    uint temp = o_ind;
    for (int i = 0; i < L_ndims; ++i) {
        uint adder = temp / L_strides[i];
        if (L_ndims - i <= S_ndims && S_dims[i - (L_ndims - S_ndims)] > 1) {
            s_ind += S_strides[i - (L_ndims - S_ndims)] * adder;
        }
        temp %= L_strides[i];
    }
    output[o_ind] = matL[o_ind] + matS[s_ind];
}

template <typename spec_t>
__global__ void ele_lazy_add_kernel(const spec_t *matA, index_t *matA_stride,
                                    const spec_t *matB, index_t *matB_stride,
                                    spec_t *output, index_t *output_stride,
                                    size_t size, size_t ndim) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    size_t index_matA = 0;
    size_t index_matB = 0;
    size_t output_index = ind;
    for (index_t i = 0; i < ndim; i++) {
        index_matA += output_index / output_stride[i] * matA_stride[i];
        index_matB += output_index / output_stride[i] * matB_stride[i];
        output_index = output_index % output_stride[i];
    }
    output[ind] = matA[index_matA] + matB[index_matB];
}

int DLGpuMatrixElementwiseAdd(const DLArrayHandle matA,
                              const DLArrayHandle matB, DLArrayHandle output,
                              bool lazy_input,
                              DLStreamHandle stream_handle = NULL) {
    int dev_id = (matA->ctx).device_id;
    cudaSetDevice(dev_id);
    size_t allocatedA = matA->ndim * sizeof(uint);
    size_t allocatedB = matB->ndim * sizeof(uint);

    uint *A_strides = (uint *)malloc(allocatedA);
    uint *B_strides = (uint *)malloc(allocatedB);
    uint *A_dims = (uint *)malloc(allocatedA);
    uint *B_dims = (uint *)malloc(allocatedB);
    size_t tmp_size = 1;
    for (int i = matA->ndim - 1; i >= 0; --i) {
        A_dims[i] = matA->shape[i];
        A_strides[i] = tmp_size;
        tmp_size *= matA->shape[i];
    }
    tmp_size = 1;
    for (int i = matB->ndim - 1; i >= 0; --i) {
        B_dims[i] = matB->shape[i];
        B_strides[i] = tmp_size;
        tmp_size *= matB->shape[i];
    }
    size_t size = 1, size_A = 1, size_B = 1;
    size_A = A_strides[0] * matA->shape[0];
    size_B = B_strides[0] * matB->shape[0];
    size = size_A > size_B ? size_A : size_B;

    dim3 blocks;
    dim3 threads;
    ThreadBlock1D(threads, blocks, size);
    void *output_data = output->data;
    void *matA_data = matA->data;
    void *matB_data = matB->data;

    assert(stream_handle != NULL);
    cudaStream_t cu_stream = static_cast<cudaStream_t>(
        stream_handle ? *(cudaStream_t *)(stream_handle->handle) : NULL);

    if (lazy_input) {
        assert(size_A == size_B);
        if (is_chunk_init(dev_id) == false) {
            chunk_init(dev_id);
        }

        index_t ndim = matA->ndim;

        index_t *matA_stride =
            (index_t *)find_chunk(ndim * sizeof(index_t), dev_id);
        index_t *matB_stride =
            (index_t *)find_chunk(ndim * sizeof(index_t), dev_id);
        index_t *output_stride =
            (index_t *)find_chunk(ndim * sizeof(index_t), dev_id);

        CUDA_CALL(cudaMemcpyAsync(matA_stride, matA->stride,
                                  ndim * sizeof(index_t),
                                  cudaMemcpyHostToDevice, cu_stream));
        CUDA_CALL(cudaMemcpyAsync(matB_stride, matB->stride,
                                  ndim * sizeof(index_t),
                                  cudaMemcpyHostToDevice, cu_stream));
        CUDA_CALL(cudaMemcpyAsync(output_stride, output->stride,
                                  ndim * sizeof(index_t),
                                  cudaMemcpyHostToDevice, cu_stream));

        HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(matA->dtype, spec_t, [&]() {
            ele_lazy_add_kernel<spec_t><<<blocks, threads, 0, cu_stream>>>(
                (const spec_t *)matA_data, matA_stride,
                (const spec_t *)matB_data, matB_stride, (spec_t *)output_data,
                output_stride, size, ndim);
        });
    } else {
        if (size_A == size_B) {
            HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(matA->dtype, spec_t, [&]() {
                ele_add_kernel<spec_t><<<blocks, threads, 0, cu_stream>>>(
                    (const spec_t *)matA_data, (const spec_t *)matB_data,
                    (spec_t *)output_data, size);
            });
        } else {
            uint *gpu_stridesA = (uint *)find_chunk(allocatedA, dev_id);
            uint *gpu_dimsA = (uint *)find_chunk(allocatedA, dev_id);
            uint *gpu_stridesB = (uint *)find_chunk(allocatedB, dev_id);
            uint *gpu_dimsB = (uint *)find_chunk(allocatedB, dev_id);
            CUDA_CALL(cudaMemcpyAsync(gpu_stridesA, A_strides, allocatedA,
                                      cudaMemcpyHostToDevice, cu_stream));
            CUDA_CALL(cudaMemcpyAsync(gpu_dimsA, A_dims, allocatedA,
                                      cudaMemcpyHostToDevice, cu_stream));
            CUDA_CALL(cudaMemcpyAsync(gpu_stridesB, B_strides, allocatedB,
                                      cudaMemcpyHostToDevice, cu_stream));
            CUDA_CALL(cudaMemcpyAsync(gpu_dimsB, B_dims, allocatedB,
                                      cudaMemcpyHostToDevice, cu_stream));
            if (size_A > size_B) {
                HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
                    matA->dtype, spec_t, [&]() {
                        ele_broadcast_add_kernel<spec_t>
                            <<<blocks, threads, 0, cu_stream>>>(
                                (const spec_t *)matA_data,
                                (const spec_t *)matB_data,
                                (spec_t *)output_data, size, gpu_stridesA,
                                gpu_stridesB, gpu_dimsA, gpu_dimsB,
                                (size_t)matA->ndim, (size_t)matB->ndim);
                    });
            } else {
                HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
                    matA->dtype, spec_t, [&]() {
                        ele_broadcast_add_kernel<spec_t>
                            <<<blocks, threads, 0, cu_stream>>>(
                                (const spec_t *)matB_data,
                                (const spec_t *)matA_data,
                                (spec_t *)output_data, size, gpu_stridesB,
                                gpu_stridesA, gpu_dimsB, gpu_dimsA,
                                (size_t)matB->ndim, (size_t)matA->ndim);
                    });
            }
        }
    }
    free(A_strides);
    free(B_strides);
    free(A_dims);
    free(B_dims);
    return 0;
}

/* below is the simple version of add elementwise */
template <typename spec_t>
__global__ void ele_lazy_add_kernel_simple(const spec_t *matA,
                                           const spec_t *matB, spec_t *output,
                                           const uint *gpu_buffer, size_t size,
                                           size_t ndim) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    const uint *matA_stride = gpu_buffer;
    const uint *matB_stride = matA_stride + ndim;
    const uint *output_stride = matB_stride + ndim;
    size_t index_matA = 0;
    size_t index_matB = 0;
    size_t output_index = ind;
    for (int i = 0; i < ndim; i++) {
        index_matA += output_index / output_stride[i] * matA_stride[i];
        index_matB += output_index / output_stride[i] * matB_stride[i];
        output_index = output_index % output_stride[i];
    }
    output[ind] = matA[index_matA] + matB[index_matB];
}

int DLGpuMatrixElementwiseAddSimple(const DLArrayHandle matA,
                                    const DLArrayHandle matB,
                                    DLArrayHandle output,
                                    DLStreamHandle stream_handle = NULL) {
    size_t size = ArrSize(matA);

    dim3 blocks;
    dim3 threads;
    ThreadBlock1D(threads, blocks, size);
    void *output_data = output->data;
    void *matA_data = matA->data;
    void *matB_data = matB->data;
    assert(stream_handle != NULL);

    cudaStream_t cu_stream = *(cudaStream_t *)(stream_handle->handle);

    HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(matA->dtype, spec_t, [&]() {
        ele_add_kernel<spec_t><<<blocks, threads, 0, cu_stream>>>(
            (const spec_t *)matA_data, (const spec_t *)matB_data,
            (spec_t *)output_data, size);
    });
    return 0;
}

int DLGpuMatrixElementwiseAddLazy(const DLArrayHandle matA,
                                  const DLArrayHandle matB,
                                  DLArrayHandle output,
                                  const DLArrayHandle gpu_buf,
                                  DLStreamHandle stream_handle = NULL) {
    size_t size = ArrSize(matA);
    size_t ndim = matA->ndim;

    dim3 blocks;
    dim3 threads;
    ThreadBlock1D(threads, blocks, size);
    void *output_data = output->data;
    void *matA_data = matA->data;
    void *matB_data = matB->data;
    const uint *gpu_buffer = (const uint *)gpu_buf->data;
    assert(stream_handle != NULL);
    cudaStream_t cu_stream = *(cudaStream_t *)(stream_handle->handle);
    HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(matA->dtype, spec_t, [&]() {
        ele_lazy_add_kernel_simple<spec_t><<<blocks, threads, 0, cu_stream>>>(
            (const spec_t *)matA_data, (const spec_t *)matB_data,
            (spec_t *)output_data, gpu_buffer, size, ndim);
    });

    return 0;
}
