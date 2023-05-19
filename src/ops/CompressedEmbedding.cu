#include "gpu_runtime.h"

__global__ void robe_hash_kernel(const int *input, const int *random_numbers,
                                 int *output, int length, int dim, int Z,
                                 bool use_slot_coef, int nslot, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    size_t iind = ind / dim;
    int npart = dim / Z;
    // (Ah*e + Bh*x + Ch*c + Dh) mod P mod |M|
    // Bh*x + Dh
    int64_t result =
        (int64_t)random_numbers[3] * input[iind] + random_numbers[1];
    // Ch*c
    result =
        result + ind % npart + (int64_t)random_numbers[2] * (ind % dim / npart);
    if (use_slot_coef) {
        // Ah*e
        result = result + (int64_t)random_numbers[4] * (iind % nslot);
    }
    // mod P mod |M|
    result = result % random_numbers[0] % length;
    output[ind] = (int)result;
}

__global__ void robe_sign_kernel(const int *input, const int *random_numbers,
                                 float *output, int dim, bool use_slot_coef,
                                 int nslot, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    size_t iind = ind / dim;
    // ((Ag*e + Bg*x + Cg*i + Dg) mod P mod 2) * 2 - 1
    // Bg*x + Dg
    int64_t result =
        (int64_t)random_numbers[7] * input[iind] + random_numbers[5];
    // Cg*i
    result = result + (int64_t)random_numbers[6] * (ind % dim);
    if (use_slot_coef) {
        // Ah*e
        result = result + (int64_t)random_numbers[8] * (iind % nslot);
    }
    // mod P mod 2
    result = result % random_numbers[0] % 2;
    result = 2 * result - 1;
    output[ind] = (float)result;
}

__global__ void mod_hash_kernel(const int *input, int *output, int nembed,
                                size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    output[ind] = input[ind] % nembed;
}

__global__ void mod_hash_negative_kernel(const int *input, int *output,
                                         int nembed, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    int prev = input[ind];
    prev = -(prev + 1);
    if (prev >= 0) {
        output[ind] = prev % nembed;
    } else {
        output[ind] = prev;
    }
}

__global__ void div_hash_kernel(const int *input, int *output, int nembed,
                                size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    output[ind] = input[ind] / nembed;
}

__global__ void compo_hash_kernel(const int *input, int *output, int ntable,
                                  int nembed, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    int *dst_ptr = output + ntable * ind;
    int ori_value = input[ind];
    for (size_t i = 0; i < ntable; ++i) {
        dst_ptr[i] = ori_value % nembed;
        ori_value /= nembed;
    }
}

__global__ void learn_hash_kernel(const int *input, const int *slope,
                                  const int *bias, const int *prime,
                                  float *output, int nbucket, int nhash,
                                  bool normal, float eps, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    size_t output_ind = ind * 2;
    size_t input_ind = output_ind / nhash;
    size_t other_ind = output_ind % nhash;
    int res0 = (input[input_ind] * slope[other_ind] + bias[other_ind])
               % prime[other_ind] % nbucket;
    float scale_pos0 = (float)res0 / (nbucket - 1);
    int res1 = (input[input_ind] * slope[other_ind + 1] + bias[other_ind + 1])
               % prime[other_ind + 1] % nbucket;
    float scale_pos1 = (float)res1 / (nbucket - 1);
    float scale_both0, scale_both1;
    if (normal) {
        float lcontent = sqrt(-2 * log(max(scale_pos0, eps)));
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

int DLGpuRobeHash(const DLArrayHandle input, const DLArrayHandle random_numbers,
                  DLArrayHandle output, int length, int dim, int Z,
                  bool use_slot_coef, DLStreamHandle stream_handle = NULL) {
    size_t size = ArrSize(output);

    assert(dim == output->shape[output->ndim - 1]);
    int nslot = input->shape[input->ndim - 1];

    const int *input_data = (const int *)input->data;
    const int *rand_data = (const int *)random_numbers->data;
    int *output_data = (int *)output->data;
    dim3 blocks, threads;
    ThreadBlock1D(threads, blocks, size);
    if (stream_handle)
        robe_hash_kernel<<<blocks, threads, 0,
                           *(cudaStream_t *)stream_handle->handle>>>(
            input_data, rand_data, output_data, length, dim, Z, use_slot_coef,
            nslot, size);
    else
        robe_hash_kernel<<<blocks, threads>>>(input_data, rand_data,
                                              output_data, length, dim, Z,
                                              use_slot_coef, nslot, size);
    return 0;
}

int DLGpuRobeSign(const DLArrayHandle input, const DLArrayHandle random_numbers,
                  DLArrayHandle output, int dim, bool use_slot_coef,
                  DLStreamHandle stream_handle = NULL) {
    size_t size = ArrSize(output);

    assert(dim == output->shape[output->ndim - 1]);
    int nslot = input->shape[input->ndim - 1];

    const int *input_data = (const int *)input->data;
    const int *rand_data = (const int *)random_numbers->data;
    float *output_data = (float *)output->data;
    dim3 blocks, threads;
    ThreadBlock1D(threads, blocks, size);
    if (stream_handle)
        robe_sign_kernel<<<blocks, threads, 0,
                           *(cudaStream_t *)stream_handle->handle>>>(
            input_data, rand_data, output_data, dim, use_slot_coef, nslot,
            size);
    else
        robe_sign_kernel<<<blocks, threads>>>(input_data, rand_data,
                                              output_data, dim, use_slot_coef,
                                              nslot, size);
    return 0;
}

int DLGpuModHash(const DLArrayHandle input, DLArrayHandle output, int nembed,
                 DLStreamHandle stream_handle = NULL) {
    size_t size = 1;
    for (index_t i = 0; i < input->ndim; i++) {
        size *= input->shape[i];
    }
    const int *input_data = (const int *)input->data;
    int *output_data = (int *)output->data;
    dim3 blocks, threads;
    ThreadBlock1D(threads, blocks, size);
    if (stream_handle)
        mod_hash_kernel<<<blocks, threads, 0,
                          *(cudaStream_t *)stream_handle->handle>>>(
            input_data, output_data, nembed, size);
    else
        mod_hash_kernel<<<blocks, threads>>>(input_data, output_data, nembed,
                                             size);
    return 0;
}

int DLGpuModHashNegative(const DLArrayHandle input, DLArrayHandle output,
                         int nembed, DLStreamHandle stream_handle = NULL) {
    size_t size = ArrSize(input);
    const int *input_data = (const int *)input->data;
    int *output_data = (int *)output->data;
    dim3 blocks, threads;
    ThreadBlock1D(threads, blocks, size);
    if (stream_handle)
        mod_hash_negative_kernel<<<blocks, threads, 0,
                                   *(cudaStream_t *)stream_handle->handle>>>(
            input_data, output_data, nembed, size);
    else
        mod_hash_negative_kernel<<<blocks, threads>>>(input_data, output_data,
                                                      nembed, size);
    return 0;
}

int DLGpuDivHash(const DLArrayHandle input, DLArrayHandle output, int nembed,
                 DLStreamHandle stream_handle = NULL) {
    size_t size = ArrSize(input);
    const int *input_data = (const int *)input->data;
    int *output_data = (int *)output->data;
    dim3 blocks, threads;
    ThreadBlock1D(threads, blocks, size);
    if (stream_handle)
        div_hash_kernel<<<blocks, threads, 0,
                          *(cudaStream_t *)stream_handle->handle>>>(
            input_data, output_data, nembed, size);
    else
        div_hash_kernel<<<blocks, threads>>>(input_data, output_data, nembed,
                                             size);
    return 0;
}

int DLGpuCompoHash(const DLArrayHandle input, DLArrayHandle output, int ntable,
                   int nembed, DLStreamHandle stream_handle = NULL) {
    size_t size = 1;
    for (index_t i = 0; i < input->ndim; i++) {
        size *= input->shape[i];
    }
    const int *input_data = (const int *)input->data;
    int *output_data = (int *)output->data;
    dim3 blocks, threads;
    ThreadBlock1D(threads, blocks, size);
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
                   DLArrayHandle output, int nbucket, bool normal, float eps,
                   DLStreamHandle stream_handle = NULL) {
    size_t size = 1;
    for (index_t i = 0; i < input->ndim; i++) {
        size *= input->shape[i];
    }
    size_t num_hash = slope->shape[0];
    size = size * num_hash / 2;
    const int *input_data = (const int *)input->data;
    const int *slope_data = (const int *)slope->data;
    const int *bias_data = (const int *)bias->data;
    const int *prime_data = (const int *)prime->data;
    float *output_data = (float *)output->data;
    dim3 blocks, threads;
    ThreadBlock1D(threads, blocks, size);
    if (stream_handle)
        learn_hash_kernel<<<blocks, threads, 0,
                            *(cudaStream_t *)stream_handle->handle>>>(
            input_data, slope_data, bias_data, prime_data, output_data, nbucket,
            num_hash, normal, eps, size);
    else
        learn_hash_kernel<<<blocks, threads>>>(
            input_data, slope_data, bias_data, prime_data, output_data, nbucket,
            num_hash, normal, eps, size);
    return 0;
}
