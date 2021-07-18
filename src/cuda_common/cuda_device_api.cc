/*!
 *  Copyright (c) 2017 by Contributors
 * \file cuda_device_api.cc
 * \brief GPU specific API
 */

#include "cuda_device_api.h"
#include <cassert>
#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CALL(cmd)                                                         \
    do {                                                                       \
        cudaError_t e = cmd;                                                   \
        if (e != cudaSuccess) {                                                \
            printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__,      \
                   cudaGetErrorString(e));                                     \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

namespace hetusys { namespace runtime {

static void GPUCopy(const void *from, void *to, size_t size,
                    cudaMemcpyKind kind, cudaStream_t stream) {
    if (stream != 0) {
        CUDA_CALL(cudaMemcpyAsync(to, from, size, kind, stream));
    } else {
        CUDA_CALL(cudaMemcpy(to, from, size, kind));
    }
}

void *CUDADeviceAPI::AllocDataSpace(DLContext ctx, size_t size,
                                    size_t alignment) {
    CUDA_CALL(cudaSetDevice(ctx.device_id));
    assert((256 % alignment) == 0U); // << "CUDA space is aligned at 256 bytes";
    void *ret;
    CUDA_CALL(cudaMalloc(&ret, size));
    return ret;
}

void CUDADeviceAPI::FreeDataSpace(DLContext ctx, void *ptr) {
    CUDA_CALL(cudaSetDevice(ctx.device_id));
    CUDA_CALL(cudaFree(ptr));
}

void CUDADeviceAPI::CopyDataFromTo(const void *from, void *to, size_t size,
                                   DLContext ctx_from, DLContext ctx_to,
                                   DLStreamHandle stream) {
    cudaStream_t cu_stream = static_cast<cudaStream_t>(
        stream ? *(cudaStream_t *)(stream->handle) : NULL);
    if (ctx_from.device_type == kGPU && ctx_to.device_type == kGPU) {
        CUDA_CALL(cudaSetDevice(ctx_from.device_id));
        if (ctx_from.device_id == ctx_to.device_id) {
            GPUCopy(from, to, size, cudaMemcpyDeviceToDevice, cu_stream);
        } else {
            cudaMemcpyPeerAsync(to, ctx_to.device_id, from, ctx_from.device_id,
                                size, cu_stream);
        }
    } else if (ctx_from.device_type == kGPU && ctx_to.device_type == kCPU) {
        CUDA_CALL(cudaSetDevice(ctx_from.device_id));
        GPUCopy(from, to, size, cudaMemcpyDeviceToHost, cu_stream);
    } else if (ctx_from.device_type == kCPU && ctx_to.device_type == kGPU) {
        CUDA_CALL(cudaSetDevice(ctx_to.device_id));
        GPUCopy(from, to, size, cudaMemcpyHostToDevice, cu_stream);
    } else {
        std::cerr << "expect copy from/to GPU or between GPU" << std::endl;
    }
}

void CUDADeviceAPI::StreamSync(DLContext ctx, DLStreamHandle stream) {
    CUDA_CALL(cudaSetDevice(ctx.device_id));
    cudaStreamSynchronize(*(cudaStream_t *)(stream->handle));
}

}} // namespace hetusys::runtime
