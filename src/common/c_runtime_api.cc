/*!
 *  Copyright (c) 2017 by Contributors
 * \file c_runtime_api.cc
 * \brief Device specific implementations
 */
#include "c_runtime_api.h"
#include "cpu_device_api.h"
#include "cuda_device_api.h"
#include "runtime_base.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>

namespace hetusys { namespace runtime {
class DeviceAPIManager {
public:
    static const int kMaxDeviceAPI = 8;
    // Get API
    static DeviceAPI *Get(DLContext ctx) {
        return Global()->GetAPI(ctx.device_type);
    }

private:
    std::array<DeviceAPI *, kMaxDeviceAPI> api_;
    DeviceAPIManager() {
        std::fill(api_.begin(), api_.end(), nullptr);
        static CPUDeviceAPI cpu_device_api_inst;
        api_[kCPU] = static_cast<DeviceAPI *>(&cpu_device_api_inst);
#ifdef DEVICE_GPU
        static CUDADeviceAPI gpu_device_api_inst;
        api_[kGPU] = static_cast<DeviceAPI *>(&gpu_device_api_inst);
#endif
    }
    // Get global static variable.
    static DeviceAPIManager *Global() {
        static DeviceAPIManager inst;
        return &inst;
    }
    // Get API.
    DeviceAPI *GetAPI(DLDeviceType type) {
        if (api_[type] == nullptr) {
            std::cout << "Device type:" << type << std::endl;
            std::cerr << "Device API not supported" << std::endl;
            exit(EXIT_FAILURE);
        }
        return api_[type];
    }
};

inline DLArray *DLArrayCreate_() {
    DLArray *arr = new DLArray();
    arr->shape = nullptr;
    arr->ndim = 0;
    arr->data = nullptr;
    return arr;
}

inline void DLArrayFree_(DLArray *arr) {
    if (arr != nullptr) {
        // ok to delete nullptr
        delete[] arr->shape;
        delete[] arr->stride;
        if (arr->data != nullptr) {
            DeviceAPIManager::Get(arr->ctx)->FreeDataSpace(arr->ctx, arr->data);
        }
    }
    delete arr;
}

inline size_t GetDataSize(DLArray *arr) {
    size_t size = 1;
    for (index_t i = 0; i < arr->ndim; ++i) {
        size *= arr->shape[i];
    }
    size *= (arr->nbits / 8);
    return size;
}

inline size_t GetDataAlignment(DLArray *arr) {
    // assume 32-bit float
    return 8;
}

}} // namespace hetusys::runtime

using namespace hetusys::runtime;

int DLArrayAlloc(const index_t *shape, const index_t *stride, index_t ndim,
                 DLContext ctx, DLArrayHandle *out, int nbits) {
    DLArray *arr = nullptr;
    API_BEGIN();
    // shape
    arr = DLArrayCreate_();
    // ndim
    arr->ndim = ndim;
    arr->nbits = nbits;
    index_t *shape_copy = new index_t[ndim];
    std::copy(shape, shape + ndim, shape_copy);
    arr->shape = shape_copy;
    index_t *stride_copy = new index_t[ndim];
    std::copy(stride, stride + ndim, stride_copy);
    arr->stride = stride_copy;
    // ctx
    arr->ctx = ctx;
    size_t size = GetDataSize(arr);
    size_t alignment = GetDataAlignment(arr);
    arr->data =
        DeviceAPIManager::Get(ctx)->AllocDataSpace(ctx, size, alignment);
    *out = arr;
    API_END_HANDLE_ERROR(DLArrayFree_(arr));
}

int DLArrayFree(DLArrayHandle handle) {
    API_BEGIN();
    DLArray *arr = handle;
    DLArrayFree_(arr);
    API_END();
}

int DLArrayCopyFromTo(DLArrayHandle from, DLArrayHandle to,
                      DLStreamHandle stream) {
    API_BEGIN();
    size_t from_size = GetDataSize(from);
    size_t to_size = GetDataSize(to);
    // The size must exactly match
    assert(from_size == to_size);
    DLContext ctx = from->ctx;
    if (ctx.device_type == kCPU) {
        ctx = to->ctx;
    } else {
        // Can not copy across different ctx types directly
        assert((to->ctx.device_type == kCPU)
               || (to->ctx.device_type == from->ctx.device_type));
    }
    DeviceAPIManager::Get(ctx)->CopyDataFromTo(from->data, to->data, from_size,
                                               from->ctx, to->ctx, stream);
    API_END();
}

int DLArrayCopyFromToOffset(DLArrayHandle from, size_t foffset,
                            DLArrayHandle to, size_t toffset, size_t copy_size,
                            DLStreamHandle stream) {
    API_BEGIN();
    DLContext ctx = from->ctx;
    if (ctx.device_type == kCPU) {
        ctx = to->ctx;
    } else {
        // Can not copy across different ctx types directly
        assert((to->ctx.device_type == kCPU)
               || (to->ctx.device_type == from->ctx.device_type));
    }
    DeviceAPIManager::Get(ctx)->CopyDataFromTo(from->data + foffset,
                                               to->data + toffset, copy_size,
                                               from->ctx, to->ctx, stream);
    API_END();
}
