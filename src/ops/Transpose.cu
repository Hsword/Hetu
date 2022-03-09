#include "gpu_runtime.h"
#include <assert.h>

#if defined(__CUDACC__)
#define OF_DEVICE_FUNC __device__ __host__ __forceinline__
#else
#define OF_DEVICE_FUNC inline
#endif

#define CUDA_1D_KERNEL_LOOP(i, n)                                                                 \
  for (int32_t i = blockIdx.x * blockDim.x + threadIdx.x, step = blockDim.x * gridDim.x; i < (n); \
       i += step)

#define CUDA_1D_KERNEL_LOOP_T(type, i, n)                                                      \
  for (type i = blockIdx.x * blockDim.x + threadIdx.x, step = blockDim.x * gridDim.x; i < (n); \
       i += step)
       
constexpr size_t kMaxMovementSize = 16;
constexpr size_t kMaxNumDims = 8;
constexpr int32_t kMov4TileSize = 32;
constexpr int32_t kMov2TileSize = 64;
constexpr int32_t kBlockRows = 8;
const int32_t kCudaThreadsNumPerBlock = 512;
const int32_t kCudaMaxBlocksNum = 8192;

inline int32_t BlocksNum4ThreadsNum(const int32_t n) {
  assert(n > 0);
  return std::min((n + kCudaThreadsNumPerBlock - 1) / kCudaThreadsNumPerBlock, kCudaMaxBlocksNum);
}

template<typename T, int N>
class NdIndexOffsetHelper {
 public:
  NdIndexOffsetHelper() {}
  template<class... Ts>
  OF_DEVICE_FUNC explicit NdIndexOffsetHelper(T d0, Ts... dims) {
    constexpr int n = 1 + sizeof...(dims);
    static_assert(n <= N, "");
    T dims_arr[n] = {d0, static_cast<T>(dims)...};
    InitStrides(dims_arr, n);
  }

  OF_DEVICE_FUNC explicit NdIndexOffsetHelper(const T* dims) { InitStrides(dims, N); }

  template<typename U>
  OF_DEVICE_FUNC explicit NdIndexOffsetHelper(const U* dims) {
    T dims_arr[N];
    for (int i = 0; i < N; ++i) { dims_arr[i] = dims[i]; }
    InitStrides(dims_arr, N);
  }

  OF_DEVICE_FUNC explicit NdIndexOffsetHelper(const T* dims, int n) { InitStrides(dims, n); }

  template<typename U>
  OF_DEVICE_FUNC explicit NdIndexOffsetHelper(const U* dims, int n) {
    T dims_arr[N];
    for (int i = 0; i < N; ++i) {
      if (i < n) { dims_arr[i] = dims[i]; }
    }
    InitStrides(dims_arr, n);
  }

  ~NdIndexOffsetHelper() = default;

  OF_DEVICE_FUNC T NdIndexToOffset(const T* index) const {
    T offset = 0;
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
    for (int i = 0; i < N - 1; ++i) { offset += index[i] * stride_[i]; }
    offset += index[N - 1];
    return offset;
  }

  OF_DEVICE_FUNC T NdIndexToOffset(const T* index, int n) const {
    assert(n <= N);
    T offset = 0;
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
    for (int i = 0; i < N; ++i) {
      if (i < n) { offset += index[i] * stride_[i]; }
    }
    return offset;
  }

  template<class... Ts>
  OF_DEVICE_FUNC T NdIndexToOffset(T d0, Ts... others) const {
    constexpr int n = 1 + sizeof...(others);
    static_assert(n <= N, "");
    T index[n] = {d0, others...};
    T offset = 0;
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
    for (int i = 0; i < n - 1; ++i) { offset += index[i] * stride_[i]; }
    if (n == N) {
      offset += index[n - 1];
    } else {
      offset += index[n - 1] * stride_[n - 1];
    }
    return offset;
  }

  OF_DEVICE_FUNC void OffsetToNdIndex(T offset, T* index) const {
    T remaining = offset;
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
    for (int i = 0; i < N - 1; ++i) {
      const T idx = remaining / stride_[i];
      index[i] = idx;
      remaining = remaining - idx * stride_[i];
    }
    index[N - 1] = remaining;
  }

  OF_DEVICE_FUNC void OffsetToNdIndex(T offset, T* index, int n) const {
    assert(n <= N);
    T remaining = offset;
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
    for (int i = 0; i < N; ++i) {
      if (i < n) {
        const T idx = remaining / stride_[i];
        index[i] = idx;
        remaining = remaining - idx * stride_[i];
      }
    }
  }

  template<class... Ts>
  OF_DEVICE_FUNC void OffsetToNdIndex(T offset, T& d0, Ts&... others) const {
    constexpr int n = 1 + sizeof...(others);
    static_assert(n <= N, "");
    T* index[n] = {&d0, &others...};
    T remaining = offset;
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
    for (int i = 0; i < n - 1; ++i) {
      const T idx = remaining / stride_[i];
      *index[i] = idx;
      remaining = remaining - idx * stride_[i];
    }
    if (n == N) {
      *index[n - 1] = remaining;
    } else {
      *index[n - 1] = remaining / stride_[n - 1];
    }
  }

  OF_DEVICE_FUNC constexpr int Size() const { return N; }

 private:
  OF_DEVICE_FUNC void InitStrides(const T* dims, const int n) {
    for (int i = n - 1; i < N; ++i) { stride_[i] = 1; }
    for (int i = n - 2; i >= 0; --i) { stride_[i] = dims[i + 1] * stride_[i + 1]; }
  }

  T stride_[N];
};

template<size_t num_dims, typename IndexType>
struct PermuteKernelParams {
  NdIndexOffsetHelper<IndexType, num_dims> src_index_helper;
  NdIndexOffsetHelper<IndexType, num_dims> dst_index_helper;
  int permutation[num_dims]{};
  IndexType count{};
  const void* src{};
  void* dst{};
};

template<size_t num_dims, typename IndexType>
PermuteKernelParams<num_dims, IndexType> MakePermuteParams(const int64_t* src_dims, const void* src,
                                                           const int* permutation, void* dst,
                                                           size_t count) {
  PermuteKernelParams<num_dims, IndexType> params;
  params.src_index_helper = NdIndexOffsetHelper<IndexType, num_dims>(src_dims);
  int64_t dst_dims[num_dims];
  for (size_t i = 0; i < num_dims; ++i) { dst_dims[i] = src_dims[permutation[i]]; }
  params.dst_index_helper = NdIndexOffsetHelper<IndexType, num_dims>(dst_dims);
  for (size_t i = 0; i < num_dims; ++i) { params.permutation[i] = permutation[i]; }
  params.src = src;
  params.dst = dst;
  params.count = static_cast<IndexType>(count);
  return params;
}

template<size_t num_dims, size_t movement_size, typename IndexType>
__global__ void PermuteKernel(PermuteKernelParams<num_dims, IndexType> params) {
  using T = typename std::aligned_storage<movement_size, movement_size>::type;
  const T* src = reinterpret_cast<const T*>(params.src);
  T* dst = reinterpret_cast<T*>(params.dst);
  IndexType src_index[num_dims];
  IndexType dst_index[num_dims];
  CUDA_1D_KERNEL_LOOP_T(IndexType, i, params.count) {
    params.dst_index_helper.OffsetToNdIndex(i, dst_index);
#pragma unroll
    for (size_t dim = 0; dim < num_dims; ++dim) {
      src_index[params.permutation[dim]] = dst_index[dim];
    }
    IndexType src_offset = params.src_index_helper.NdIndexToOffset(src_index);
    dst[i] = src[src_offset];
  }
}

// (B, X, Y) -> (B, Y, X)
// refer from https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/
template<size_t num_dims, size_t movement_size, size_t tile_size, typename IndexType>
__global__ void BatchTransposeKernel(const void* src_ptr, void* dst_ptr, IndexType rows,
                                     IndexType cols, IndexType num_tile_rows,
                                     IndexType num_tile_cols, int32_t block_nums) {
  const IndexType src_rows = rows;
  const IndexType src_cols = cols;
  const IndexType dst_rows = cols;
  const IndexType dst_cols = rows;

  using T = typename std::aligned_storage<movement_size, movement_size>::type;
  __shared__ T tile[tile_size][tile_size + 1];  // To avoid bank conflict.

  const T* src = reinterpret_cast<const T*>(src_ptr);
  T* dst = reinterpret_cast<T*>(dst_ptr);

  IndexType batch_num_tile = num_tile_rows * num_tile_cols;
  for (int i = blockIdx.x, step = gridDim.x; i < block_nums; i += step) {
    const IndexType batch_index = i / batch_num_tile;  // the index of batch.
    const IndexType tile_index =
        i - batch_index * batch_num_tile;  // equal to i % (num_tile_rows*num_tile_cols). the
                                           // flatten index of tile in a batch.

    const IndexType tile_row_index =
        tile_index / num_tile_cols;  // the row index of tile in a batch.
    const IndexType tile_col_index =
        tile_index
        - tile_row_index
              * num_tile_cols;  // equal to k % num_tile_cols. the col index of tile in a batch.

    const IndexType offset = batch_index * src_rows * src_cols;
    {
      IndexType col_in_tile = threadIdx.x;
      IndexType col_in_matrix = tile_col_index * tile_size + threadIdx.x;
#pragma unroll
      for (IndexType row_in_tile = threadIdx.y; row_in_tile < tile_size;
           row_in_tile += kBlockRows) {
        IndexType row_in_matrix = row_in_tile + tile_row_index * tile_size;
        if (col_in_matrix < src_cols && row_in_matrix < src_rows) {
          tile[row_in_tile][col_in_tile] = src[offset + row_in_matrix * src_cols + col_in_matrix];
        }
      }
    }
    __syncthreads();
    {
      IndexType col_in_tile = threadIdx.x;
      IndexType col_in_matrix = tile_row_index * tile_size + threadIdx.x;
#pragma unroll
      for (IndexType row_in_tile = threadIdx.y; row_in_tile < tile_size;
           row_in_tile += kBlockRows) {
        IndexType row_in_matrix = row_in_tile + tile_col_index * tile_size;
        if (col_in_matrix < dst_cols && row_in_matrix < dst_rows) {
          dst[offset + row_in_matrix * dst_cols + col_in_matrix] = tile[col_in_tile][row_in_tile];
        }
      }
    }
    __syncthreads();
  }
}

/*
Here is a Movementsie=2 version of Batch Transpose.
When the H W can be divided by 2. we can read data use movementsize=4, and write back as
movementsize=4.
*/
template<size_t num_dims, size_t tile_size, typename IndexType>
__global__ void BatchTransposeMovement2Kernel(const void* src_ptr, void* dst_ptr, IndexType rows,
                                              IndexType cols, IndexType num_tile_rows,
                                              IndexType num_tile_cols, int32_t block_nums) {
  const IndexType src_rows = rows;
  const IndexType src_cols = cols;
  const IndexType dst_rows = cols;
  const IndexType dst_cols = rows;

  static_assert(tile_size % 2 == 0, "");
  using T_MOV2 = typename std::aligned_storage<2, 2>::type;
  using T_MOV4 = typename std::aligned_storage<4, 4>::type;

  const T_MOV4* src = reinterpret_cast<const T_MOV4*>(src_ptr);
  T_MOV4* dst = reinterpret_cast<T_MOV4*>(dst_ptr);

  // Use union structure to process Load and Store.
  __shared__ union {
    T_MOV2 tile_m2[tile_size][tile_size + 2];      // half [64][66]
    T_MOV4 tile_m4[tile_size][tile_size / 2 + 1];  // half2 [64][33]
  } tile_mem;

  IndexType batch_num_tile = num_tile_rows * num_tile_cols;
  for (int i = blockIdx.x, step = gridDim.x; i < block_nums; i += step) {
    const IndexType batch_index = i / batch_num_tile;  // the index of batch.
    const IndexType tile_index =
        i - batch_index * batch_num_tile;  // equal to i % (num_tile_rows*num_tile_cols). the
                                           // flatten index of tile in a batch.

    const IndexType tile_row_index =
        tile_index / num_tile_cols;  // the row index of tile in a batch.
    const IndexType tile_col_index =
        tile_index
        - tile_row_index
              * num_tile_cols;  // equal to k % num_tile_cols. the col index of tile in a batch.

    const IndexType offset = batch_index * src_rows * src_cols;
    {
      IndexType col_in_tile = threadIdx.x;
      IndexType col_in_matrix = tile_col_index * tile_size + threadIdx.x * 2;
#pragma unroll
      for (IndexType row_in_tile = threadIdx.y; row_in_tile < tile_size;
           row_in_tile += kBlockRows) {
        IndexType row_in_matrix = row_in_tile + tile_row_index * tile_size;
        if (col_in_matrix < src_cols && row_in_matrix < src_rows) {
          tile_mem.tile_m4[row_in_tile][col_in_tile] =
              src[(offset + row_in_matrix * src_cols + col_in_matrix) / 2];
        }
      }
    }
    __syncthreads();
    {
      IndexType col_in_tile = threadIdx.x;
      IndexType col_in_matrix = tile_row_index * tile_size + threadIdx.x * 2;
#pragma unroll
      for (IndexType row_in_tile = threadIdx.y; row_in_tile < tile_size;
           row_in_tile += kBlockRows) {
        IndexType row_in_matrix = row_in_tile + tile_col_index * tile_size;
        union {
          T_MOV4 m4;
          T_MOV2 m2[2];
        } tmp_storage;

        if (col_in_matrix < dst_cols && row_in_matrix < dst_rows) {
          tmp_storage.m2[0] = tile_mem.tile_m2[col_in_tile * 2][row_in_tile];
          tmp_storage.m2[1] = tile_mem.tile_m2[col_in_tile * 2 + 1][row_in_tile];
          dst[(offset + row_in_matrix * dst_cols + col_in_matrix) / 2] = tmp_storage.m4;
        }
      }
    }
    __syncthreads();
  }
}

template<size_t max_movement_size>
size_t GetMovementSize(size_t elem_size, size_t num_dims, const int64_t* src_dims, const void* src,
                       const int* permutation, void* dst) {
    static_assert(max_movement_size > 0 && (max_movement_size & (max_movement_size - 1)) == 0, "");
    assert(elem_size > 0);
    assert((elem_size & (elem_size - 1)) == 0);
    assert((max_movement_size % elem_size) == 0);

    if (permutation[num_dims - 1] == num_dims - 1) {
        const int64_t last_dim_size = src_dims[num_dims - 1] * elem_size;
        auto src_ptr = reinterpret_cast<std::uintptr_t>(src);
        auto dst_ptr = reinterpret_cast<std::uintptr_t>(dst);
        for (size_t size = max_movement_size; size > elem_size; size /= 2) {
            if (last_dim_size % size == 0 && src_ptr % size == 0 && dst_ptr % size == 0) { return size; }
        }
    }
    return elem_size;
}

template<size_t max_num_dims>
void SimplifyPermutation(size_t num_dims, const int64_t* src_dims, const int* permutation,
                         size_t* simplified_num_dims, int64_t* simplified_src_dims,
                         int* simplified_permutation) {
    assert(num_dims > 0);
    int64_t coalesced_dims[max_num_dims];
    size_t start_permutation_index = 0;
    while (start_permutation_index < num_dims) {
        const size_t start_dim_index = permutation[start_permutation_index];
        coalesced_dims[start_dim_index] = src_dims[start_dim_index];
        size_t end_permutation_index = start_permutation_index + 1;
        while (end_permutation_index < num_dims
               && permutation[end_permutation_index] == permutation[end_permutation_index - 1] + 1) {
            const size_t end_dim_index = permutation[end_permutation_index];
            coalesced_dims[start_dim_index] *= src_dims[end_dim_index];
            coalesced_dims[end_dim_index] = 1;
            end_permutation_index += 1;
        }  
        start_permutation_index = end_permutation_index;
    }
    size_t valid_num_dims = 0;
    int mapping[max_num_dims];
    for (size_t i = 0; i < num_dims; ++i) {
        const int src_dim = coalesced_dims[i];
        if (src_dim == 1) {
            mapping[i] = -1;
        } else {
            mapping[i] = valid_num_dims;
            simplified_src_dims[valid_num_dims] = src_dim;
            valid_num_dims += 1;
        }
    }
  if (valid_num_dims == 0) {
    *simplified_num_dims = 1;
    simplified_src_dims[0] = 1;
    simplified_permutation[0] = 0;
  } else {
    *simplified_num_dims = valid_num_dims;
    size_t permutation_index = 0;
    for (size_t i = 0; i < num_dims; ++i) {
      const int mapped = mapping[permutation[i]];
      if (mapped >= 0) {
        simplified_permutation[permutation_index] = mapped;
        permutation_index += 1;
      }
    }
  }
}

template<size_t max_num_dims, size_t max_movement_size>
void SimplifyPermutation(size_t num_dims, const int64_t* src_dims, const int* permutation,
                         size_t* simplified_num_dims, int64_t* simplified_src_dims,
                         int* simplified_permutation, size_t elem_size, const void* src, void* dst,
                         size_t* movement_size) {
  const size_t pre_simplified_movement_size =
      GetMovementSize<max_movement_size>(elem_size, num_dims, src_dims, src, permutation, dst);
  int64_t tmp_dims[max_num_dims];
  for (size_t i = 0; i < num_dims; ++i) { tmp_dims[i] = src_dims[i]; }
  tmp_dims[num_dims - 1] /= (pre_simplified_movement_size / elem_size);
  SimplifyPermutation<max_num_dims>(num_dims, tmp_dims, permutation, simplified_num_dims,
                                    simplified_src_dims, simplified_permutation);
  *movement_size =
      GetMovementSize<max_movement_size>(pre_simplified_movement_size, *simplified_num_dims,
                                         simplified_src_dims, src, simplified_permutation, dst);
  simplified_src_dims[*simplified_num_dims - 1] /= (*movement_size / pre_simplified_movement_size);
}

template<size_t num_dims, size_t movement_size, size_t tile_size, typename IndexType>
void LaunchBatchTransposeKernel(cudaStream_t& cuda_stream,
                                const PermuteKernelParams<num_dims, IndexType>& params,
                                const IndexType& num_batches, const IndexType& rows,
                                const IndexType& cols) {
  IndexType num_tile_rows = (rows + tile_size - 1) / tile_size;
  IndexType num_tile_cols = (cols + tile_size - 1) / tile_size;
  const int32_t block_nums = num_batches * num_tile_rows * num_tile_cols;
  int32_t launched_block_nums = std::min(block_nums, kCudaMaxBlocksNum);
  if (tile_size == kMov2TileSize) {
    const int32_t half2_thread = tile_size / 2;  // cause each thread process two half elements.
    BatchTransposeMovement2Kernel<num_dims, kMov2TileSize, IndexType>
        <<<launched_block_nums, dim3(half2_thread, kBlockRows), 0, cuda_stream>>>(
            params.src, params.dst, rows, cols, num_tile_rows, num_tile_cols,
            block_nums);  // Set threads num as 32x8 cause each threads
                          // process 4 elements to 64x66 half share memory.
  } else {
    BatchTransposeKernel<num_dims, movement_size, tile_size, IndexType>
        <<<launched_block_nums, dim3(tile_size, kBlockRows), 0, cuda_stream>>>(
            params.src, params.dst, rows, cols, num_tile_rows, num_tile_cols, block_nums);
  }
}

template<size_t tile_size, typename IndexType>
bool CheckIfGreaterEqualThanTileSize(const IndexType& rows, const IndexType& cols) {
  if (rows < tile_size || cols < tile_size) { return false; }
  return true;
}

template<size_t num_dims, size_t tile_size, typename IndexType>
bool CheckLaunchBatchTranspose(const int* permutation, const IndexType& num_batches,
                               const IndexType& rows, const IndexType& cols) {
  if (CheckIfGreaterEqualThanTileSize<tile_size, IndexType>(rows, cols)) {
    if (num_batches == 1 && permutation[1] == 0 && permutation[0] == 1) {
      // 2d tensor case: (0, 1) -> (1, 0)
      return true;
    } else if (num_dims == 3 && permutation[2] == 1 && permutation[1] == 2) {
      // 3d tensor case: (0, 1, 2) -> (0, 2, 1)
      return true;
    } else {
      return false;
    }
  }
  return false;
}

template<typename IndexType, size_t movement_size>
bool CheckUseMov2(const IndexType& rows, const IndexType& cols, const void* src, void* dst) {
  auto src_ptr = reinterpret_cast<std::uintptr_t>(src);
  auto dst_ptr = reinterpret_cast<std::uintptr_t>(dst);
  return (movement_size == 2) && (rows % 2 == 0) && (cols % 2 == 0) && (src_ptr % 4 == 0)
         && (dst_ptr % 4 == 0);
}

template<size_t num_dims, typename IndexType>
void InferBatchTransposeShape(const int64_t* src_dims, IndexType* num_batches, IndexType* rows,
                              IndexType* cols) {
  if (num_dims == 2) {
    *num_batches = 1;
    *rows = src_dims[0];
    *cols = src_dims[1];
  } else {
    *num_batches = src_dims[0];
    *rows = src_dims[1];
    *cols = src_dims[2];
  }
}

template<size_t num_dims, size_t movement_size, typename IndexType>
void LaunchKernel(cudaStream_t cuda_stream, const int64_t* src_dims, const void* src, const int* permutation,
                  void* dst, size_t count) {
  PermuteKernelParams<num_dims, IndexType> params =
      MakePermuteParams<num_dims, IndexType>(src_dims, src, permutation, dst, count);

  if (num_dims == 2 || num_dims == 3) {
    IndexType num_batches;
    IndexType rows;
    IndexType cols;
    InferBatchTransposeShape<num_dims, IndexType>(src_dims, &num_batches, &rows, &cols);
    if (CheckLaunchBatchTranspose<num_dims, kMov4TileSize>(params.permutation, num_batches, rows,
                                                           cols)) {
      if (CheckUseMov2<IndexType, movement_size>(rows, cols, src, dst)) {
        LaunchBatchTransposeKernel<num_dims, 2, kMov2TileSize, IndexType>(cuda_stream, params,
                                                                          num_batches, rows, cols);
      } else {
        LaunchBatchTransposeKernel<num_dims, movement_size, kMov4TileSize, IndexType>(
            cuda_stream, params, num_batches, rows, cols);
      }
    } else {
      PermuteKernel<num_dims, movement_size, IndexType>
          <<<BlocksNum4ThreadsNum(params.count), kCudaThreadsNumPerBlock, 0, cuda_stream>>>(params);
    }
  } else {
    PermuteKernel<num_dims, movement_size, IndexType>
        <<<BlocksNum4ThreadsNum(params.count), kCudaThreadsNumPerBlock, 0, cuda_stream>>>(params);
  }
}

template<size_t num_dims, size_t movement_size>
void DispatchIndexType(cudaStream_t stream, const int64_t* src_dims, const void* src,
                       const int* permutation, void* dst) {
  size_t count = 1;
  for (size_t i = 0; i < num_dims; ++i) { count *= src_dims[i]; }  
  LaunchKernel<num_dims, movement_size, int32_t>(stream, src_dims, src, permutation, dst, count);
}

template<size_t num_dims>
void DispatchMovementSize(cudaStream_t stream, size_t movement_size, const int64_t* src_dims,
                          const void* src, const int* permutation, void* dst) {
  void (*func)(cudaStream_t /*stream*/, const int64_t* /*src_dims*/, const void* /*src*/,
               const int* /*permutation*/, void* /*dst*/) = nullptr;
  if (movement_size == 1) {
    func = DispatchIndexType<num_dims, 1>;
  } else if (movement_size == 2) {
    func = DispatchIndexType<num_dims, 2>;
  } else if (movement_size == 4) {
    func = DispatchIndexType<num_dims, 4>;
  } else if (movement_size == 8) {
    func = DispatchIndexType<num_dims, 8>;
  } else if (movement_size == 16) {
    func = DispatchIndexType<num_dims, 16>;
  } else {
    assert(false);
  }
  func(stream, src_dims, src, permutation, dst);
}


void LaunchWithSimplified(cudaStream_t stream, size_t movement_size, size_t num_dims,
                          const int64_t* src_dims, const void* src, const int* permutation,
                          void* dst) {
  void (*func)(cudaStream_t /*stream*/, size_t /*movement_size*/, const int64_t* /*src_dims*/,
               const void* /*src*/, const int* /*permutation*/, void* /*dst*/) = nullptr;
  if (num_dims == 1) {
    func = DispatchMovementSize<1>;
  } else if (num_dims == 2) {
    func = DispatchMovementSize<2>;
  } else if (num_dims == 3) {
    func = DispatchMovementSize<3>;
  } else if (num_dims == 4) {
    func = DispatchMovementSize<4>;
  } else if (num_dims == 5) {
    func = DispatchMovementSize<5>;
  } else if (num_dims == 6) {
    func = DispatchMovementSize<6>;
  } else if (num_dims == 7) {
    func = DispatchMovementSize<7>;
  } else if (num_dims == 8) {
    func = DispatchMovementSize<8>;
  } else {
    assert(false);
  }
  func(stream, movement_size, src_dims, src, permutation, dst);
}

int DLGpuTranspose(const DLArrayHandle input, DLArrayHandle output,
                       const int *permutation, const int64_t *src_dims, DLStreamHandle stream_handle = NULL) {
    const float *src = (const float *)input->data;
    float *dst = (float *)output->data;
    const size_t num_dims = output->ndim;

    size_t simplified_num_dims = 0;
    int64_t simplified_src_dims[kMaxNumDims];
    int simplified_permutation[kMaxNumDims];
    size_t movement_size = 0;
    
    SimplifyPermutation<kMaxNumDims, kMaxMovementSize>(
      num_dims, src_dims, permutation, &simplified_num_dims, simplified_src_dims,
      simplified_permutation, 4, src, dst, &movement_size);
      
    cudaStream_t cu_stream = NULL;
    if(stream_handle)
      cu_stream = (*(cudaStream_t *)(stream_handle->handle));
    LaunchWithSimplified(cu_stream, movement_size, simplified_num_dims, simplified_src_dims, src,
                       simplified_permutation, dst);
    return 0;
}