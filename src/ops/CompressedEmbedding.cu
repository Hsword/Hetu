#include "gpu_runtime.h"

__global__ void mod_hash_kernel(const float *input, float *output, int nembed,
                                size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    output[ind] = (float)((int)input[ind] % nembed);
}

__global__ void compo_hash_kernel(const float *input, float *output, int ntable,
                                  int nembed, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    float *dst_ptr = output + ntable * ind;
    int ori_value = (int)input[ind];
    for (size_t i = 0; i < ntable; ++i) {
        dst_ptr[i] = (float)(ori_value % nembed);
        ori_value /= nembed;
    }
}

__global__ void learn_hash_kernel(const float *input, const float *slope,
                                  const float *bias, const float *prime,
                                  float *output, int nbucket, int nhash,
                                  bool normal, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    size_t output_ind = ind * 2;
    size_t input_ind = output_ind / nhash;
    size_t other_ind = output_ind % nhash;
    int res0 = (int)(input[input_ind] * slope[other_ind] + bias[other_ind])
               % (int)prime[other_ind] % nbucket;
    float scale_pos0 = (float)res0 / (nbucket - 1);
    int res1 =
        (int)(input[input_ind] * slope[other_ind + 1] + bias[other_ind + 1])
        % (int)prime[other_ind + 1] % nbucket;
    float scale_pos1 = (float)res1 / (nbucket - 1);
    float scale_both0, scale_both1;
    if (normal) {
        float lcontent = sqrt(-2 * log(scale_pos0));
        float rcontent = 2 * scale_pos1;
        scale_both0 = lcontent * cospi(rcontent);
        scale_both1 = lcontent * sinpi(rcontent);
    } else {
        scale_both0 = scale_pos0 * 2 - 1;
        scale_both1 = scale_pos1 * 2 - 1;
    }
    output[output_ind] = scale_both0;
    output[output_ind + 1] = scale_both1;
}

int DLGpuModHash(const DLArrayHandle input, DLArrayHandle output, int nembed,
                 DLStreamHandle stream_handle = NULL) {
    size_t size = 1;
    for (index_t i = 0; i < input->ndim; i++) {
        size *= input->shape[i];
    }
    dim3 blocks;
    dim3 threads;
    const float *input_data = (const float *)input->data;
    float *output_data = (float *)output->data;
    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }
    if (stream_handle)
        mod_hash_kernel<<<blocks, threads, 0,
                          *(cudaStream_t *)stream_handle->handle>>>(
            input_data, output_data, nembed, size);
    else
        mod_hash_kernel<<<blocks, threads>>>(input_data, output_data, nembed,
                                             size);
    return 0;
}

int DLGpuCompoHash(const DLArrayHandle input, DLArrayHandle output, int ntable,
                   int nembed, DLStreamHandle stream_handle = NULL) {
    size_t size = 1;
    for (index_t i = 0; i < input->ndim; i++) {
        size *= input->shape[i];
    }
    dim3 blocks;
    dim3 threads;
    const float *input_data = (const float *)input->data;
    float *output_data = (float *)output->data;
    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }
    if (stream_handle)
        compo_hash_kernel<<<blocks, threads, 0,
                            *(cudaStream_t *)stream_handle->handle>>>(
            input_data, output_data, ntable, nembed, size);
    else
        compo_hash_kernel<<<blocks, threads>>>(input_data, output_data, ntable,
                                               nembed, size);
    return 0;
}

int DLGpuLearnHash(const DLArrayHandle input, const DLArrayHandle slope,
                   const DLArrayHandle bias, const DLArrayHandle prime,
                   DLArrayHandle output, int nbucket, bool normal,
                   DLStreamHandle stream_handle = NULL) {
    size_t size = 1;
    for (index_t i = 0; i < input->ndim; i++) {
        size *= input->shape[i];
    }
    size_t num_hash = slope->shape[0];
    size = size * num_hash / 2;
    dim3 blocks;
    dim3 threads;
    const float *input_data = (const float *)input->data;
    const float *slope_data = (const float *)slope->data;
    const float *bias_data = (const float *)bias->data;
    const float *prime_data = (const float *)prime->data;
    float *output_data = (float *)output->data;
    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }
    if (stream_handle)
        learn_hash_kernel<<<blocks, threads, 0,
                            *(cudaStream_t *)stream_handle->handle>>>(
            input_data, slope_data, bias_data, prime_data, output_data, nbucket,
            num_hash, normal, size);
    else
        learn_hash_kernel<<<blocks, threads>>>(
            input_data, slope_data, bias_data, prime_data, output_data, nbucket,
            num_hash, normal, size);
    return 0;
}
