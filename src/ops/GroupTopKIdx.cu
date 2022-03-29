#include "gpu_runtime.h"

__global__ void grouptopkidx_kernel(float* input, float* top1_group, float* output_idx, int k, int num_local_gpus, int col){
    int row = blockIdx.x * blockDim.x  + threadIdx.x;
    float *start_input_val = input + row*col;
    float *start_output_idx = output_idx + row*k;
    int group = (int)top1_group[row];
    float tmp_val;
    float tmp_idx;
    float cache[2048];//k<100
    for(int i=0; i<k; i++){
        tmp_val=-10000.0;
        tmp_idx=-1;
        for(int j=group*num_local_gpus; j<(group+1)*num_local_gpus; j++){
            if(start_input_val[j] >= tmp_val){
                tmp_val=start_input_val[j];
                tmp_idx=j;
            }
        }
        cache[i]=tmp_val;
        start_output_idx[i]=tmp_idx;
        start_input_val[(int)tmp_idx]=-10000.0;
    }
    for(int i=0;i<k;i++){
        start_input_val[int(start_output_idx[i])]=cache[i];
    }
}

int DLGpuGroupTopKIdx(const DLArrayHandle input, DLArrayHandle top1_group, DLArrayHandle output_idx, int k, int num_local_gpus, DLStreamHandle stream_handle){
    assert(output_idx->ndim == 2); 
    assert(input->ndim == 2); // In MoE we need to find the topk of 2-d tensor on the last dimention
    int ROW = input->shape[0];
    assert(output_idx->shape[0]==ROW);
    int COL = input->shape[1];
    assert(k >= 1);
    int num_local_experts = COL/num_local_gpus;
    assert(k <= num_local_experts);
    assert(output_idx->shape[1]==k);
    assert(top1_group->ndim==2);
    assert(top1_group->shape[0]==ROW);
    assert(top1_group->shape[1]==1);


    float* input_data = (float*) input->data;
    float* top1_group_data = (float*) top1_group->data;
    float* output_idx_data = (float*) output_idx->data;

    dim3 blocks;
    dim3 threads;

    if(ROW<=1024){
        blocks.x = 1;
        threads.x = ROW;
    }else{
        blocks.x = (ROW+1023)/1024;
        threads.x = 1024;
    }
    
    if(stream_handle){
        grouptopkidx_kernel<<<blocks, threads, 0, *(cudaStream_t *)stream_handle->handle>>>(
                input_data, top1_group_data, output_idx_data, k, num_local_gpus, COL
                );
    }else{
        grouptopkidx_kernel<<<blocks, threads>>>(
                input_data, top1_group_data, output_idx_data, k, num_local_gpus, COL
                );
    }
    return 0;
}
