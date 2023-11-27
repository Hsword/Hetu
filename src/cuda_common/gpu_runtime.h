#ifndef HETUSYS_SRC_GPU_RUNTIME_H
#define HETUSYS_SRC_GPU_RUNTIME_H

#include "../common/c_runtime_api.h"
#include <iostream>
#include <vector>
#include <set>
#include <map>
#include <cassert>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cfloat>
#include <cudnn.h>
#include <stack>
#include <cusparse.h>
#include <cusparse_v2.h>
#include <curand.h>

#define THREADS_PER_BLOCK 1024

#define CUDNN_CALL(cmd)                                                        \
    do {                                                                       \
        cudnnStatus_t e = cmd;                                                 \
        if (e != CUDNN_STATUS_SUCCESS) {                                       \
            printf("Failed: Cudnn error %s:%d '%s'\n", __FILE__, __LINE__,     \
                   cudnnGetErrorString(e));                                    \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define CUBLAS_CALL(func)                                                      \
    {                                                                          \
        cublasStatus_t err = (func);                                           \
        assert(err == CUBLAS_STATUS_SUCCESS);                                  \
    }

#define CUDA_CALL(cmd)                                                         \
    do {                                                                       \
        cudaError_t e = cmd;                                                   \
        if (e != cudaSuccess) {                                                \
            printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__,      \
                   cudaGetErrorString(e));                                     \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define CUSP_CALL(func)                                                        \
    {                                                                          \
        cusparseStatus_t e = (func);                                           \
        assert((e == CUSPARSE_STATUS_SUCCESS));                                \
    }
// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n)                                                 \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);               \
         i += blockDim.x * gridDim.x)

static const char *curandGetErrorString(curandStatus_t status) {
    switch (status) {
    case CURAND_STATUS_SUCCESS:
        return "CURAND_STATUS_SUCCESS";
    case CURAND_STATUS_VERSION_MISMATCH:
        return "CURAND_STATUS_VERSION_MISMATCH";
    case CURAND_STATUS_NOT_INITIALIZED:
        return "CURAND_STATUS_NOT_INITIALIZED";
    case CURAND_STATUS_ALLOCATION_FAILED:
        return "CURAND_STATUS_ALLOCATION_FAILED";
    case CURAND_STATUS_TYPE_ERROR:
        return "CURAND_STATUS_TYPE_ERROR";
    case CURAND_STATUS_OUT_OF_RANGE:
        return "CURAND_STATUS_OUT_OF_RANGE";
    case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
        return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
    case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
        return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
    case CURAND_STATUS_LAUNCH_FAILURE:
        return "CURAND_STATUS_LAUNCH_FAILURE";
    case CURAND_STATUS_PREEXISTING_FAILURE:
        return "CURAND_STATUS_PREEXISTING_FAILURE";
    case CURAND_STATUS_INITIALIZATION_FAILED:
        return "CURAND_STATUS_INITIALIZATION_FAILED";
    case CURAND_STATUS_ARCH_MISMATCH:
        return "CURAND_STATUS_ARCH_MISMATCH";
    case CURAND_STATUS_INTERNAL_ERROR:
        return "CURAND_STATUS_INTERNAL_ERROR";
    }
    return "Unknown cuRAND error";
}

#define CURAND_CALL(cmd)                                                       \
    do {                                                                       \
        curandStatus_t e = cmd;                                                \
        if (e != CURAND_STATUS_SUCCESS) {                                      \
            printf("Failed: Curand error %s:%d '%s'\n", __FILE__, __LINE__,    \
                   curandGetErrorString(e));                                   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

void DebugCudaMalloc(cudaError_t tmp);

// extern struct Chunk;
struct Chunk {
    void *ptr;
    size_t chunk_size;
    Chunk() {
        ptr = NULL;
        chunk_size = 0;
    }
    Chunk(void *_ptr, size_t _chunk_size) {
        ptr = _ptr;
        chunk_size = _chunk_size;
    }
    bool operator<(const Chunk &tmp) const {
        if (chunk_size != tmp.chunk_size)
            return chunk_size < tmp.chunk_size;
        else if (ptr != tmp.ptr)
            return ptr < tmp.ptr;
        else {
            return false;
        }
    }
};

extern std::map<int, bool> init_free_chunk_set;
extern std::map<int, std::multiset<Chunk>> free_chunk_set;

bool is_chunk_init(size_t dev_id = 0);
void chunk_init(size_t dev_id = 0);
void del_chunk(void *ptr, size_t dev_id = 0);
void *find_chunk(size_t _chunk_size, size_t dev_id = 0, bool debug = true);

extern std::map<size_t, bool> is_cudnn_init;
extern std::map<size_t, cudnnHandle_t> cudnn_map;
void cudnn_init(size_t dev_id = 0, DLStreamHandle stream = NULL);

extern std::map<size_t, bool> is_cusp_init;
extern std::map<size_t, cusparseHandle_t> cusp_map;
void cusp_init(size_t dev_id = 0, DLStreamHandle stream = NULL);

extern std::map<size_t, bool> is_cublas_init;
extern std::map<size_t, cublasHandle_t> cublas_map;
void cublas_init(size_t dev_id = 0, DLStreamHandle stream = NULL);

void cuda_init();
int GetThreadNum(int cnt);

void ThreadBlock1D(dim3 &threads, dim3 &blocks, size_t size);
void ThreadBlock2D(dim3 &threads, dim3 &blocks, size_t xsize, size_t ysize);
size_t ArrSize(const DLArrayHandle array);

#endif
