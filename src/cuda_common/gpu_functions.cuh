#ifndef HETUSYS_SRC_GPU_FUNCTIONS_CUH
#define HETUSYS_SRC_GPU_FUNCTIONS_CUH

#include "random.h"
#include <nppdefs.h>
#include <curand_kernel.h>

template <class T>
inline __device__ float __numeric_limits_on_device() {
    __builtin_unreachable();
}

template <>
inline __device__ float __numeric_limits_on_device<uint8_t>() {
    return (float)NPP_MAX_8U;
}

template <>
inline __device__ float __numeric_limits_on_device<uint16_t>() {
    return (float)NPP_MAX_16U;
}

template <class T>
inline __device__ T stochastic_rounding(float input, float scale, float minele,
                                        HetuRandomState &cudars, size_t ind) {
    float result = (input - minele) / scale;
    curandStatePhilox4_32_10_t state;
    curand_init(cudars.seed, cudars.seqnum, ind, &state);
    result += curand_uniform(&state);
    result = max(result, 0.0);
    result = min(result, __numeric_limits_on_device<T>());
    return (T)floorf(result);
}

template <class T>
inline __device__ T fixed_rounding(float input, float scale, float minele) {
    float result = (input - minele) / scale;
    result += 0.5;
    result = max(result, 0.0);
    result = min(result, __numeric_limits_on_device<T>());
    return (T)floorf(result);
}

#endif
