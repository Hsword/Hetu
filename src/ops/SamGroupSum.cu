#include "gpu_runtime.h"

__global__ void sam_group_sum_kernel(float* gate_data, float* output_data, int num_local_gpus, int col){
    int row = blockIdx.x * blockDim.x  + threadIdx.x;
    float *start_input_val = gate_data + row*col;
    float *start_output_data = output_data + row*num_local_gpus;
    int num_local_experts = col/num_local_gpus;
    float sum=0;
    for(int i=0; i<num_local_gpus; i++){
        sum = 0;
        for(int j=i*num_local_experts; j<(i+1)*num_local_experts; j++){
            sum += start_input_val[j];
        }
        start_output_data[i]=sum;
    }
}

int DLGpuSamGroupSum(const DLArrayHandle gate, DLArrayHandle output, int num_local_gpus, 
        DLStreamHandle stream_handle){
    assert(output->ndim == 2); 
    assert(gate->ndim == 2); // In MoE we need to find the topk of 2-d tensor on the last dimention
    int ROW = gate->shape[0];
    assert(output->shape[0]==ROW);
    int COL = gate->shape[1];
    assert(COL%num_local_gpus == 0);
    assert(output->shape[1]==num_local_gpus);

    float* gate_data = (float*) gate->data;
    float* output_data = (float*) output->data;

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
        sam_group_sum_kernel<<<blocks, threads, 0, *(cudaStream_t *)stream_handle->handle>>>(
                gate_data, output_data, num_local_gpus, COL
                );
    }else{
        sam_group_sum_kernel<<<blocks, threads>>>(
                gate_data, output_data, num_local_gpus, COL
                );
    }
    return 0;
}
