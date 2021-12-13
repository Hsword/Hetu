#include "gpu_runtime.h"

__global__ void max_kernel(float* input, float* output_idx, float* output_val, int ROW, int COL){
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    if (col >= COL){
        return;
    }
    float tmp = -1e12;
    float tmp_idx = 0;
    for(int i=0;i<ROW;i++){
        if(input[col+i*COL]>tmp){
            tmp=input[col+i*COL];
            tmp_idx=i;
        }
    }
    output_idx[col] = tmp_idx;
    output_val[col] = tmp;
}

int DLGpuMax(const DLArrayHandle input, DLArrayHandle output_idx, DLArrayHandle output_val, int dim, DLStreamHandle stream_handle=NULL){
    assert(dim==0);// only support dim=0 now
    float* input_data=(float*)input->data;
    float* output_idx_data=(float*)output_idx->data;
    float* output_val_data=(float*)output_val->data;

    dim3 blocks;
    dim3 threads;
    
    int ROW = input->shape[0];
    int COL = input->shape[1];

    if(COL<=1024){
        blocks.x=1;
        threads.x=COL;
    }else{
        blocks.x = (COL+1023)/1024;
        threads.x = 1024;
    }
    

    if(stream_handle){
        max_kernel<<<blocks, threads, 0, *(cudaStream_t *)stream_handle->handle>>>(input_data, output_idx_data, output_val_data, ROW, COL);
    }else{
        max_kernel<<<blocks, threads>>>(input_data, output_idx_data, output_val_data, ROW, COL);
    }

    return 0;







}
