#include "gpu_runtime.h"

__device__ inline float euclidean_dist(float a, float b) {
    float diff = a - b;
    return diff * diff;
}

__device__ inline float innerproduct_dist(float a, float b) {
    return a * b;
}

__global__ void min_dist_kernel(const float *lookups, const float *key,
                                float *codebook, float *indices, float *output,
                                size_t numkey, size_t length, bool mode,
                                size_t size) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size)
        return;
    const float *src_ptr = lookups + length * index;
    int min_id = -1;
    float min_dist = -1;
    for (int i = 0; i < numkey; ++i) {
        float cur_dist = 0;
        const float *dst_ptr = key + length * i;
        for (int j = 0; j < length; ++j) {
            if (mode) {
                cur_dist += euclidean_dist(src_ptr[j], dst_ptr[j]);
            } else {
                cur_dist += innerproduct_dist(src_ptr[j], dst_ptr[j]);
            }
        }
        if (cur_dist < min_dist || min_dist < 0) {
            min_dist = cur_dist;
            min_id = i;
        }
    }
    size_t code_index = indices[index];
    codebook[code_index] = min_id;
    const float *in_ptr = key + length * min_id;
    float *out_ptr = output + length * index;
    for (int j = 0; j < length; ++j) {
        out_ptr[j] = in_ptr[j];
    }
}

int DLGpuMinDist(const DLArrayHandle lookup, const DLArrayHandle key,
                 DLArrayHandle codebook, DLArrayHandle indices,
                 DLArrayHandle output, bool mode,
                 DLStreamHandle stream_handle = NULL) {
    size_t size = 1;
    for (int i = 0; i < indices->ndim; i++) {
        size *= indices->shape[i];
    }
    size_t numkey = key->shape[0];
    size_t length = key->shape[1];
    dim3 blocks;
    dim3 threads;
    const float *lookup_data = (const float *)lookup->data;
    const float *key_data = (const float *)key->data;
    float *code_data = (float *)codebook->data;
    float *ind_data = (float *)indices->data;
    float *out_data = (float *)output->data;
    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }
    if (stream_handle)
        min_dist_kernel<<<blocks, threads, 0,
                          *(cudaStream_t *)stream_handle->handle>>>(
            lookup_data, key_data, code_data, ind_data, out_data, numkey,
            length, mode, size);
    else
        min_dist_kernel<<<blocks, threads>>>(lookup_data, key_data, code_data,
                                             ind_data, out_data, numkey, length,
                                             mode, size);
    return 0;
}
