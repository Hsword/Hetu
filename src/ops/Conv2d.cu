#include "gpu_runtime.h"

extern __global__ void array_set_kernel(float *output, float value,
                                        size_t size);
extern __global__ void float_memory_copy(float *A, const float *B, size_t len);
extern int Float_Add(float *A, const float *B, int len,
                     DLStreamHandle stream_handle);

__global__ void im2col_kernel(int N, int C, int H, int W, int filter_outChannel,
                              int filter_H, int filter_W,
                              const float *input_data_x, float *workspace_data,
                              const int padding, const int stride,
                              const int blocks) {
    int block_id = blockIdx.x;
    int thread_id = threadIdx.x;
    int max_threads_per_block = blockDim.x;
    int thread_index = block_id * max_threads_per_block + thread_id;
    int out_H = (H + 2 * padding - filter_H) / stride + 1;
    int out_W = (W + 2 * padding - filter_W) / stride + 1;
    for (int i = thread_index; i < N * C * out_H * out_W;
         i += blocks * max_threads_per_block) {
        int N_i = i / (C * out_H * out_W);
        int base_N = N_i * C * out_H * out_W;
        int C_i = (i - base_N) / (out_H * out_W);
        int base_C = C_i * out_H * out_W;
        int out_H_i = (i - base_N - base_C) / out_W;
        int out_W_i = i % out_W;
        assert(base_N + base_C + out_H_i * out_W + out_W_i == i);
        int in_x = out_H_i * stride - padding;
        int in_y = out_W_i * stride - padding;
        for (int x = in_x; x < in_x + filter_H; x++)
            for (int y = in_y; y < in_y + filter_W; y++) {
                if (x < 0 || x >= H || y < 0 || y >= W)
                    workspace_data[(base_N + base_C) * filter_H * filter_W
                                   + ((x - in_x) * filter_W + (y - in_y))
                                         * out_H * out_W
                                   + out_H_i * out_W + out_W_i] = 0;
                else
                    workspace_data[(base_N + base_C) * filter_H * filter_W
                                   + ((x - in_x) * filter_W + (y - in_y))
                                         * out_H * out_W
                                   + out_H_i * out_W + out_W_i] =
                        input_data_x[(N_i * C + C_i) * H * W + x * W + y];
            }
    }
}

__global__ void gemm_kernel(const float *A, const float *B, float *C, int rowA,
                            int colA, int rowB, int colB) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    assert(rowB % colA == 0);
    int K = rowB / colA;
    if (r >= rowA || c >= colB)
        return;
    for (int k = 0; k < K; k++) {
        float Cvalue = 0.0;
        for (int e = 0; e < colA; e++)
            Cvalue += A[r * colA + e] * B[(e + k * colA) * colB + c];
        C[(r + k * rowA) * colB + c] = Cvalue;
    }
}

int DLGpuConv2d(const DLArrayHandle input_x, const DLArrayHandle input_f,
                DLArrayHandle output, DLArrayHandle workspace_arr,
                const int padding, const int stride,
                DLStreamHandle stream_handle = NULL) {
    assert(input_x->ndim == 4);
    assert(input_f->ndim == 4);
    assert(input_x->shape[1] == input_f->shape[1]);
    int N = input_x->shape[0];
    int C = input_x->shape[1];
    int H = input_x->shape[2];
    int W = input_x->shape[3];
    int filter_outChannel = input_f->shape[0];
    // int filter_inChannel = input_f->shape[1];
    int filter_H = input_f->shape[2];
    int filter_W = input_f->shape[3];
    assert((H + 2 * padding - filter_H) % stride == 0);
    assert((W + 2 * padding - filter_W) % stride == 0);
    int out_H = (H + 2 * padding - filter_H) / stride + 1;
    int out_W = (W + 2 * padding - filter_W) / stride + 1;
    int y_col_size = out_H * out_W;
    int y_row_size = C * filter_H * filter_W;

    const float *input_data_x = (const float *)input_x->data;
    const float *input_data_f = (const float *)input_f->data;
    float *output_data = (float *)output->data;
    float *workspace_data = (float *)workspace_arr->data;
    // get max threads and blocks
    int dev_id = (input_x->ctx).device_id;
    ;
    cudaSetDevice(dev_id);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev_id);
    int threads = deviceProp.maxThreadsPerBlock;
    int blocks = deviceProp.maxThreadsPerMultiProcessor / threads
                 * deviceProp.multiProcessorCount;
    // im2col kernel
    if (stream_handle)
        im2col_kernel<<<blocks, threads, 0,
                        *(cudaStream_t *)stream_handle->handle>>>(
            N, C, H, W, filter_outChannel, filter_H, filter_W, input_data_x,
            workspace_data, padding, stride, blocks);
    else
        im2col_kernel<<<blocks, threads>>>(
            N, C, H, W, filter_outChannel, filter_H, filter_W, input_data_x,
            workspace_data, padding, stride, blocks);
    // sgemm
    const int BLOCK_SIZE = 16;
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((std::max(y_row_size, y_col_size) + dimBlock.x - 1)
                     / dimBlock.x,
                 (std::max(filter_outChannel, y_row_size) + dimBlock.y - 1)
                     / dimBlock.y);
    if (stream_handle)
        gemm_kernel<<<dimGrid, dimBlock, 0,
                      *(cudaStream_t *)stream_handle->handle>>>(
            input_data_f, workspace_data, output_data, filter_outChannel,
            y_row_size, N * y_row_size, y_col_size);
    else
        gemm_kernel<<<dimGrid, dimBlock>>>(
            input_data_f, workspace_data, output_data, filter_outChannel,
            y_row_size, N * y_row_size, y_col_size);
    return 0;
}
__global__ void trans_im2col_kernel(int N, int C, int H, int W,
                                    int filter_outChannel, int filter_H,
                                    int filter_W, float *input_data_x,
                                    float *workspace_data, const int padding,
                                    const int stride, const int blocks) {
    int block_id = blockIdx.x;
    int thread_id = threadIdx.x;
    int max_threads_per_block = blockDim.x;
    int thread_index = block_id * max_threads_per_block + thread_id;
    int out_H = (H + 2 * padding - filter_H) / stride + 1;
    int out_W = (W + 2 * padding - filter_W) / stride + 1;
    for (int i = thread_index; i < N * C * out_H * out_W;
         i += blocks * max_threads_per_block) {
        int N_i = i / (C * out_H * out_W);
        int base_N = N_i * C * out_H * out_W;
        int C_i = (i - base_N) / (out_H * out_W);
        int base_C = C_i * out_H * out_W;
        int out_H_i = (i - base_N - base_C) / out_W;
        int out_W_i = i % out_W;
        assert(base_N + base_C + out_H_i * out_W + out_W_i == i);
        int in_x = out_H_i * stride - padding;
        int in_y = out_W_i * stride - padding;
        for (int x = in_x; x < in_x + filter_H; x++)
            for (int y = in_y; y < in_y + filter_W; y++) {
                if (x < 0 || x >= H || y < 0 || y >= W)
                    workspace_data[(base_N + base_C) * filter_H * filter_W
                                   + ((x - in_x) * filter_W + (y - in_y))
                                         * out_H * out_W
                                   + out_H_i * out_W + out_W_i] = 0;
                else
                    atomicAdd(
                        &input_data_x[(N_i * C + C_i) * H * W + x * W + y],
                        workspace_data[(base_N + base_C) * filter_H * filter_W
                                       + ((x - in_x) * filter_W + (y - in_y))
                                             * out_H * out_W
                                       + out_H_i * out_W + out_W_i]);
            }
    }
}
__global__ void transA_gemm_kernel(const float *A, const float *B, float *C,
                                   int rowA, int colA, int rowB, int colB) {
    size_t r = blockIdx.x * blockDim.x + threadIdx.x;
    size_t c = blockIdx.y * blockDim.y + threadIdx.y;
    if (r >= colA || c >= colB)
        return;
    assert(rowB % rowA == 0);
    size_t batch_size = rowB / rowA;
    // output shape(output_batch, filter_col_size, output_col_size)
    for (int i = 0; i < batch_size; i++) {
        float tmp = 0;
        // C[batch_size][colA][colB]  -> C[i][r][c]
        for (int j = 0; j < rowA; j++)
            // A[j][r] * B[i][j][c]
            tmp += A[j * colA + r] * B[i * rowA * colB + j * colB + c];
        C[i * colA * colB + r * colB + c] = tmp;
    }
}
__global__ void batch_transB_gemm_kernel(const float *A, const float *B,
                                         float *C, int rowA, int colA, int rowB,
                                         int colB, int batch_size) {
    size_t r = blockIdx.x * blockDim.x + threadIdx.x;
    size_t c = blockIdx.y * blockDim.y + threadIdx.y;
    if (r >= rowA || c >= rowB)
        return;
    assert(colA == colB);
    // output shape(batch_size, filter_row_size, filter_col_size)
    for (int i = 0; i < batch_size; i++) {
        float tmp = 0;
        // C[batch_size][rowA][rowB]  -> C[i][r][c]
        for (int j = 0; j < colA; j++)
            // A[i][r][j] * B[i][c][j]
            tmp += A[i * rowA * colB + r * colB + j]
                   * B[i * rowB * colB + c * colB + j];
        C[i * rowA * rowB + r * rowB + c] = tmp;
    }
}

int DLGpuConv2d_Gradient_of_Data(const DLArrayHandle input_f,
                                 const DLArrayHandle gradient_y,
                                 DLArrayHandle gradient_x,
                                 DLArrayHandle workspace_im2col,
                                 const int padding, const int stride,
                                 DLStreamHandle stream_handle = NULL) {
    size_t input_N = gradient_x->shape[0];
    size_t input_C = gradient_x->shape[1];
    size_t input_H = gradient_x->shape[2];
    size_t input_W = gradient_x->shape[3];
    size_t filter_outChannel = input_f->shape[0];
    size_t filter_inChannel = input_f->shape[1];
    size_t filter_H = input_f->shape[2];
    size_t filter_W = input_f->shape[3];
    size_t output_N = gradient_y->shape[0];
    size_t output_C = gradient_y->shape[1];
    size_t output_H = gradient_y->shape[2];
    size_t output_W = gradient_y->shape[3];

    float *gradient_x_data = (float *)gradient_x->data;
    float *output_data = (float *)gradient_y->data;
    size_t output_batch = output_N;
    size_t output_row_size = output_C;
    size_t output_col_size = output_H * output_W;
    const float *filter_data = (const float *)input_f->data;
    size_t filter_row_size = filter_outChannel;
    size_t filter_col_size = filter_inChannel * filter_H * filter_W;

    float *gradient_im2col_XX;
    gradient_im2col_XX = (float *)workspace_im2col->data;

    // output size (output_N, filter_C * filter_H * filter_W, output_H *
    // output*W)  == (output_batch, filter_col_size, output_col_size)
    const int BLOCK_SIZE = 16;
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((filter_col_size + BLOCK_SIZE - 1) / dimBlock.x,
                 (output_col_size + BLOCK_SIZE - 1) / dimBlock.y);
    if (stream_handle)
        transA_gemm_kernel<<<dimGrid, dimBlock, 0,
                             *(cudaStream_t *)stream_handle->handle>>>(
            filter_data, output_data, gradient_im2col_XX, filter_row_size,
            filter_col_size, output_batch * output_row_size, output_col_size);
    else
        transA_gemm_kernel<<<dimGrid, dimBlock>>>(
            filter_data, output_data, gradient_im2col_XX, filter_row_size,
            filter_col_size, output_batch * output_row_size, output_col_size);
    // get max threads and blocks
    int dev_id = (input_f->ctx).device_id;
    cudaSetDevice(dev_id);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev_id);
    int threads = deviceProp.maxThreadsPerBlock;
    int blocks = deviceProp.maxThreadsPerMultiProcessor / threads
                 * deviceProp.multiProcessorCount;
    // get the gradient of input_x
    size_t numthread = input_N * input_C * input_H * input_W;
    size_t numblocks = (numthread + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (stream_handle)
        array_set_kernel<<<numblocks, THREADS_PER_BLOCK, 0,
                           *(cudaStream_t *)stream_handle->handle>>>(
            gradient_x_data, 0, numthread);
    else
        array_set_kernel<<<numblocks, THREADS_PER_BLOCK>>>(gradient_x_data, 0,
                                                           numthread);
    if (stream_handle)
        trans_im2col_kernel<<<blocks, threads, 0,
                              *(cudaStream_t *)stream_handle->handle>>>(
            input_N, input_C, input_H, input_W, filter_outChannel, filter_H,
            filter_W, gradient_x_data, gradient_im2col_XX, padding, stride,
            blocks);
    else
        trans_im2col_kernel<<<blocks, threads>>>(
            input_N, input_C, input_H, input_W, filter_outChannel, filter_H,
            filter_W, gradient_x_data, gradient_im2col_XX, padding, stride,
            blocks);

    return 0;
}

int DLGpuConv2d_Gradient_of_Filter(const DLArrayHandle input_x,
                                   const DLArrayHandle gradient_y,
                                   DLArrayHandle gradient_f,
                                   DLArrayHandle workspace_im2col,
                                   DLArrayHandle workspace_batch_filter,
                                   const int padding, const int stride,
                                   DLStreamHandle stream_handle = NULL) {
    size_t input_N = input_x->shape[0];
    size_t input_C = input_x->shape[1];
    size_t input_H = input_x->shape[2];
    size_t input_W = input_x->shape[3];
    size_t filter_outChannel = gradient_f->shape[0];
    size_t filter_inChannel = gradient_f->shape[1];
    size_t filter_H = gradient_f->shape[2];
    size_t filter_W = gradient_f->shape[3];
    size_t output_N = gradient_y->shape[0];
    size_t output_C = gradient_y->shape[1];
    size_t output_H = gradient_y->shape[2];
    size_t output_W = gradient_y->shape[3];

    const float *input_x_data = (const float *)input_x->data;
    float *gradient_f_data = (float *)gradient_f->data;
    float *output_data = (float *)gradient_y->data;
    size_t output_batch = output_N;
    size_t output_row_size = output_C;
    size_t output_col_size = output_H * output_W;
    size_t filter_row_size = filter_outChannel;
    size_t filter_col_size = filter_inChannel * filter_H * filter_W;

    // get max threads and blocks
    int dev_id = (input_x->ctx).device_id;
    cudaSetDevice(dev_id);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev_id);
    int threads = deviceProp.maxThreadsPerBlock;
    int blocks = deviceProp.maxThreadsPerMultiProcessor / threads
                 * deviceProp.multiProcessorCount;

    float *im2col_XX;
    im2col_XX = (float *)workspace_im2col->data;
    if (stream_handle)
        im2col_kernel<<<blocks, threads, 0,
                        *(cudaStream_t *)stream_handle->handle>>>(
            input_N, input_C, input_H, input_W, filter_outChannel, filter_H,
            filter_W, input_x_data, im2col_XX, padding, stride, blocks);
    else
        im2col_kernel<<<blocks, threads>>>(
            input_N, input_C, input_H, input_W, filter_outChannel, filter_H,
            filter_W, input_x_data, im2col_XX, padding, stride, blocks);

    size_t im2col_XX_row = filter_col_size;
    size_t im2col_XX_col = output_col_size;
    float *batch_filter;
    // batch_filter = new float[input_N * filter_row_size * filter_col_size];
    batch_filter = (float *)(workspace_batch_filter->data);
    const int BLOCK_SIZE = 16;
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((filter_row_size + BLOCK_SIZE - 1) / dimBlock.x,
                 (filter_col_size + BLOCK_SIZE - 1) / dimBlock.y);
    if (stream_handle)
        batch_transB_gemm_kernel<<<dimGrid, dimBlock, 0,
                                   *(cudaStream_t *)stream_handle->handle>>>(
            output_data, im2col_XX, batch_filter, output_row_size,
            output_col_size, im2col_XX_row, im2col_XX_col, output_batch);
    else
        batch_transB_gemm_kernel<<<dimGrid, dimBlock>>>(
            output_data, im2col_XX, batch_filter, output_row_size,
            output_col_size, im2col_XX_row, im2col_XX_col, output_batch);
    size_t total = filter_row_size * filter_col_size;
    while (output_batch != 1) {
        Float_Add(batch_filter, batch_filter + (output_batch + 1) / 2 * total,
                  output_batch / 2 * total, stream_handle);
        output_batch = (output_batch + 1) / 2;
    }
    size_t BLOCKS = (total + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (stream_handle)
        float_memory_copy<<<BLOCKS, THREADS_PER_BLOCK, 0,
                            *(cudaStream_t *)stream_handle->handle>>>(
            gradient_f_data, batch_filter, total);
    else
        float_memory_copy<<<BLOCKS, THREADS_PER_BLOCK>>>(gradient_f_data,
                                                         batch_filter, total);
    return 0;
}
