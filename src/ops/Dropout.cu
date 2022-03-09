#include "gpu_runtime.h"
#include <curand_kernel.h>

constexpr int32_t kVecSize = 4;
constexpr int32_t kBlockSize = 256;

template<typename T, int pack_size>
struct GetPackType {
  using type = typename std::aligned_storage<pack_size * sizeof(T), pack_size * sizeof(T)>::type;
};

template<typename T, int pack_size>
using PackType = typename GetPackType<T, pack_size>::type;

template<typename T, int pack_size>
union Pack {
  static_assert(sizeof(PackType<T, pack_size>) == sizeof(T) * pack_size, "");
  __device__ Pack() {
    // do nothing
  }
  PackType<T, pack_size> storage;
  T elem[pack_size];
};

template<typename T, int pack_size>
__device__ inline Pack<T, pack_size> FetchPack(const PackType<T, pack_size>* ptr) {
  Pack<T, pack_size> pack;
  pack.storage = *ptr;
  return pack;
}

union RandPack4 {
  float4 storage;
  float elem[4];
};

struct CUDAGeneratorState {
  uint64_t dev_offset;
  int32_t dev_counter;
};

template<typename T, int pack_size, bool tail>
__global__ void FusedDropoutAddGpu(uint64_t seed, CUDAGeneratorState* cuda_gen_state, uint64_t inc_offset,
    const int64_t elem_cnt, float rate, float scale, int64_t n_tail, const T* x,
     T* y, const T* tail_x, T* tail_y) {
  int32_t global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(seed, global_thread_id, cuda_gen_state->dev_offset, &state);
  using LoadType = PackType<T, pack_size>;
  using LoadPack = Pack<T, pack_size>;
  
  T t_scale = static_cast<T>(scale);
  RandPack4 rand_uniform_pack4;
  for (int64_t linear_index = global_thread_id * pack_size; linear_index < elem_cnt;
       linear_index += gridDim.x * blockDim.x * pack_size) {
    rand_uniform_pack4.storage = curand_uniform4(&state);
    const LoadType* x_load = reinterpret_cast<const LoadType*>(x + linear_index);
    LoadPack x_vec;
    x_vec.storage = *x_load;

    LoadPack y_vec;
#pragma unroll
    for (int i = 0; i < pack_size; i++) {
      T tmp_float_mask = static_cast<float>(rand_uniform_pack4.elem[i] > rate);
      y_vec.elem[i] = x_vec.elem[i] * tmp_float_mask * t_scale;
    }
    *(reinterpret_cast<LoadType*>(y + linear_index)) = y_vec.storage;
  }

  if (tail && global_thread_id < n_tail) {
    const float rand_uniform = curand_uniform(&state);
    const int8_t mask_val = rand_uniform > rate;
    T tmp_float_mask = static_cast<float>(mask_val);
    T tmp_tail_out = tail_x[global_thread_id] * tmp_float_mask * t_scale;
    tail_y[global_thread_id] = tmp_tail_out;
  }

  __syncthreads();
  
  if (threadIdx.x == 0) {
    int32_t new_counter = atomicAdd(&cuda_gen_state->dev_counter, 1) + 1;
    if (new_counter == gridDim.x) {
      cuda_gen_state->dev_counter = 0;           // reset counter to zero
      cuda_gen_state->dev_offset += inc_offset;  // maintain the state of generator's dev_offset
    }
  }
}

template<int pack_size>
unsigned int ComputeGridSize(cudaDeviceProp devProp, const int32_t block_size, const int64_t elem_cnt) {
    const int32_t max_threads_multi_process = devProp.maxThreadsPerMultiProcessor;
    const int32_t multi_processor_count = devProp.multiProcessorCount;
    unsigned int blocks_per_sm = max_threads_multi_process / block_size;
    unsigned int grid_size = ((elem_cnt + block_size - 1) / block_size);
    grid_size = std::min((unsigned int)multi_processor_count * blocks_per_sm, grid_size);
    return grid_size;
}

template<typename T>
void DispatchTail(cudaStream_t stream, cudaDeviceProp devProp, uint64_t seed, CUDAGeneratorState* cuda_gen_state,
                  const int64_t elem_cnt, float rate, float scale, const T* x, T* y) {
  unsigned int grid_size = ComputeGridSize<4>(devProp, kBlockSize, elem_cnt);
  constexpr int pack_size = 4;
  const int64_t pack_num = elem_cnt / pack_size;
  const int64_t tail_offset = pack_num * pack_size;
  const int64_t n_tail = elem_cnt - tail_offset;
  const bool tail = n_tail > 0 ? true : false;
  uint64_t inc_offset = 0;

  if (tail) {
    inc_offset = ((elem_cnt - 1) / (kBlockSize * grid_size * kVecSize) + 1) * kVecSize + 1;
    FusedDropoutAddGpu<T, pack_size, true><<<grid_size, kBlockSize, 0, stream>>>(
            seed, cuda_gen_state, inc_offset, elem_cnt, rate, scale, n_tail, x, y,
            (x + tail_offset), (y + tail_offset));
  } else {
    inc_offset = ((elem_cnt - 1) / (kBlockSize * grid_size * kVecSize) + 1) * kVecSize;
    FusedDropoutAddGpu<T, pack_size, false><<<grid_size, kBlockSize, 0, stream>>>(
            seed, cuda_gen_state, inc_offset, elem_cnt, rate, scale, n_tail, x, y,
            nullptr, nullptr);
  }
}

__global__ void InitCurandStatesKernel(CUDAGeneratorState* cuda_gen_state) {
  cuda_gen_state->dev_counter = static_cast<int32_t>(0);
  cuda_gen_state->dev_offset = static_cast<uint64_t>(0);
}

int DLGpuDropout(const DLArrayHandle input, const float dropout,
                 DLArrayHandle output, unsigned long long *pseed,
                 DLStreamHandle stream_handle = NULL) {
    size_t size = 1;
    for (int i = 0; i < input->ndim; i++) {
        size *= input->shape[i];
    }
    const float *input_data = (const float *)input->data;
    float *output_data = (float *)output->data;
    
    cudaStream_t cu_stream = NULL;
    if(stream_handle)
        cu_stream = (*(cudaStream_t *)(stream_handle->handle));

    const float rate = dropout;
    float scale = 0.0;
    if (rate < 1.0f) { scale = 1.0f / (1.0f - rate); }
        
    CUDAGeneratorState *cuda_gen_state;
    cudaMalloc(&cuda_gen_state, sizeof(CUDAGeneratorState));
    InitCurandStatesKernel<<<1, 1>>>(cuda_gen_state);
    cudaDeviceProp devProp;
    int dev_id = (input->ctx).device_id;
    cudaGetDeviceProperties(&devProp, dev_id);
    DispatchTail<float>(cu_stream, devProp, *pseed, cuda_gen_state, size, rate, scale, input_data, output_data);
    
    return 0;
}

__global__ void dropout_gradient_kernel(const float *grad, const float *fw_output,
                                float *output,const float rate,
                                size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    float fw_out = fw_output[ind];
    float keep_mask = (float)(fw_out > 1e-10 || fw_out < -1e-10);
    output[ind] = grad[ind] * keep_mask / (1 - rate);
}

int DLGpuDropoutGradient(const DLArrayHandle grad, const DLArrayHandle fw_output,
                         const float dropout, DLArrayHandle output,
                         DLStreamHandle stream_handle = NULL) {
    size_t size = 1;
    for (index_t i = 0; i < grad->ndim; i++) {
        size *= grad->shape[i];
    }
    const float *grad_data = (const float *)grad->data;
    const float *fw_output_data = (const float *)fw_output->data;
    float *output_data = (float *)output->data;

    dim3 blocks;
    dim3 threads;
    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }
    if (stream_handle) {
        dropout_gradient_kernel<<<blocks, threads, 0,
                         *(cudaStream_t *)stream_handle->handle>>>(
            grad_data, fw_output_data, output_data, dropout, size);
    } else {
        dropout_gradient_kernel<<<blocks, threads>>>(grad_data, fw_output_data, 
                        output_data, dropout, size);
    }
    return 0;
}
