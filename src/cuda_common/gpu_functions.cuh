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

template <class T>
inline __device__ float __signed_numeric_min_on_device() {
    __builtin_unreachable();
}

template <class T>
inline __device__ float __signed_numeric_max_on_device() {
    __builtin_unreachable();
}

template <>
inline __device__ float __signed_numeric_min_on_device<int8_t>() {
    return (float)NPP_MIN_8S;
}

template <>
inline __device__ float __signed_numeric_min_on_device<int16_t>() {
    return (float)NPP_MIN_16S;
}

template <>
inline __device__ float __signed_numeric_max_on_device<int8_t>() {
    return (float)NPP_MAX_8S;
}

template <>
inline __device__ float __signed_numeric_max_on_device<int16_t>() {
    return (float)NPP_MAX_16S;
}

template <class T>
inline __device__ T signed_stochastic_rounding(float input, float scale,
                                               float middle,
                                               HetuRandomState &cudars,
                                               size_t ind) {
    float result = (input - middle) / scale;
    curandStatePhilox4_32_10_t state;
    curand_init(cudars.seed, cudars.seqnum, ind, &state);
    result += curand_uniform(&state);
    result = max(result, __signed_numeric_min_on_device<T>());
    result = min(result, __signed_numeric_max_on_device<T>());
    return (T)floorf(result);
}

template <class T>
inline __device__ T signed_fixed_rounding(float input, float scale,
                                          float middle) {
    float result = (input - middle) / scale;
    result += 0.5;
    result = max(result, __signed_numeric_min_on_device<T>());
    result = min(result, __signed_numeric_max_on_device<T>());
    return (T)floorf(result);
}

#endif
