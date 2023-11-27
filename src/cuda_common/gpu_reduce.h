#include "gpu_runtime.h"

__forceinline__ __device__ float WarpReduceSum(float val) {
  unsigned int mask = __ballot_sync(0xFFFFFFFF, true);
  for (unsigned int k = (warpSize >> 1); k > 0; k >>= 1)
    val += __shfl_down_sync(mask, val, k, warpSize);
  return val;
}

__forceinline__ __device__ void BlockReduceSum(float &val, float *shared) {
    int tid = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = WarpReduceSum(val);

    __syncthreads();
    if (tid == 0)
        shared[wid] = val;

    __syncthreads();
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[tid] : 0;

    if (wid == 0)
        val = WarpReduceSum(val);
}

__forceinline__ __device__ void WarpReduceArgmax(float &val, size_t &ptr) {
  float tmp_val;
  size_t tmp_ptr;
  unsigned int mask = __ballot_sync(0xFFFFFFFF, true);
  for (unsigned int k = (warpSize >> 1); k > 0; k >>= 1) {
    tmp_val = __shfl_down_sync(mask, val, k, warpSize);
    tmp_ptr = __shfl_down_sync(mask, ptr, k, warpSize);
    if (ptr == SIZE_MAX || tmp_ptr == SIZE_MAX)
        continue;
    if (tmp_val > val) {
        val = tmp_val;
        ptr = tmp_ptr;
    }
    else if(tmp_val == val && tmp_ptr < ptr) {
        ptr = tmp_ptr;
    }
  }
}

__forceinline__ __device__ void BlockReduceArgmax(float &val, size_t &ptr, float *shared_value, size_t* shared_ptr) {
    int tid = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    WarpReduceArgmax(val, ptr);

    __syncthreads();
    if (tid == 0) {
        shared_value[wid] = val;
        shared_ptr[wid] = ptr;
    }

    __syncthreads();
    if (threadIdx.x < blockDim.x / warpSize) {
        val = shared_value[tid];
        ptr = shared_ptr[tid];
    }
    else {
        val = 0;
        ptr = SIZE_MAX;
    }

    if (wid == 0)
        WarpReduceArgmax(val, ptr);
}
