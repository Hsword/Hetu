#include "gpu_runtime.h"

__global__ void topkval_kernel(float* input, float* output_idx, float* output_val, int k, int col){
    int row = blockIdx.x * blockDim.x  + threadIdx.x;
    float *start_input_val = input + row*col;
    float *start_output_idx = output_idx + row*k;
    float* start_output_val = output_val + row*k;

    for(int i=0; i<k; i++){
        start_output_val[i]=start_input_val[int(start_output_idx[i])];
    }
}

int DLGpuTopKVal(const DLArrayHandle input, DLArrayHandle output_idx, DLArrayHandle output_val, int k, 
        DLStreamHandle stream_handle){
    assert(output_idx->ndim == 2); 
    assert(input->ndim == 2); // In MoE we need to find the topk of 2-d tensor on the last dimention
    int ROW = input->shape[0];
    assert(output_idx->shape[0]==ROW);
    assert(output_val->shape[0]==ROW);
    int COL = input->shape[1];
    assert(k >= 1);
    assert(k <= COL);
    assert(output_idx->shape[1]==k);

    float* input_data = (float*) input->data;
    float* output_idx_data = (float*) output_idx->data;
    float* output_val_data = (float*) output_val->data;

    dim3 blocks;
    dim3 threads;

    if(ROW<=1024){
        blocks.x = 1;
        threads.x = ROW;
    }else{
        blocks.x = (ROW+1023)/1024;
        threads.x = 1024;
    }
    
//    int* record;
//    cudaMalloc((void**)&record, sizeof(int)*ROW*COL);
//    cudaMemset(record, 0, sizeof(int)*ROW*COL);

    if(stream_handle){
        topkval_kernel<<<blocks, threads, 0, *(cudaStream_t *)stream_handle->handle>>>(
                input_data, output_idx_data, output_val_data, k, COL
                );
    }else{
        topkval_kernel<<<blocks, threads>>>(
                input_data, output_idx_data, output_val_data, k, COL
                );
    }
    return 0;
}
