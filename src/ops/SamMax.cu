#include "gpu_runtime.h"

__global__ void sammax_kernel(float* input, float* top1_group, float* topk_indice, float* output, int num_local_gpus, int col){
    int row = blockIdx.x * blockDim.x  + threadIdx.x;
    float *start_input_val = input + row*col;
    float *start_output = output + row*col;
    int start = int(top1_group[row])*num_local_gpus;
    int end = int(top1_group[row]+1)*num_local_gpus;
    float tmp = start_input_val[int(topk_indice[row])];
    for(int j=0; j<start; j++){
        if(start_input_val[j] > tmp){
            start_output[j] = start_input_val[j]-tmp;
        }else{
            start_output[j] = 0.0;
        }
    }
    
    for(int j=end; j<col; j++){
        if(start_input_val[j] > tmp){
            start_output[j] = start_input_val[j]-tmp;
        }else{
            start_output[j] = 0.0;
        }
    }
}


__global__ void sammax_grad_gate_kernel(float* output_grad, float* input, float* top1_group, float* topk_indice, float* output, int num_local_gpus, int col){
    int row = blockIdx.x * blockDim.x  + threadIdx.x;
    float *start_output_grad = output_grad+row*col;
    float *start_input_val = input + row*col;
    float *start_output = output + row*col;
    int start = int(top1_group[row])*num_local_gpus;
    int end = int(top1_group[row]+1)*num_local_gpus;
    float tmp = start_input_val[int(topk_indice[row])];
    for(int j=0; j<start; j++){
        if(start_input_val[j] > tmp){
            start_output[j] = start_output_grad[j];
        }else{
            start_output[j] = 0.0;
        }
    }
    for(int j=start; j<end; j++){
        start_output_grad[j]=0.0;
    }

    for(int j=end; j<col; j++){
        if(start_input_val[j] > tmp){
            start_output[j] = start_output_grad[j];
        }else{
            start_output[j] = 0.0;
        }
    }
}

int DLGpuSamMax(const DLArrayHandle input, DLArrayHandle top1_group, DLArrayHandle topk_indice, DLArrayHandle output, int num_local_gpus, DLStreamHandle stream_handle){
    assert(output->ndim == 2); 
    assert(input->ndim == 2); // In MoE we need to find the topk of 2-d tensor on the last dimention
    int ROW = input->shape[0];
    assert(output->shape[0]==ROW);
    int COL = input->shape[1];
    assert(output->shape[1]==COL);
    assert(top1_group->ndim==2);
    assert(top1_group->shape[0]==ROW);
    assert(top1_group->shape[1]==1);
    assert(topk_indice->ndim==2);
    assert(topk_indice->shape[0]==ROW);
    assert(topk_indice->shape[1]==1);

    float* input_data = (float*) input->data;
    float* top1_group_data = (float*)top1_group->data;
    float* topk_indice_data = (float*)topk_indice->data;
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
        sammax_kernel<<<blocks, threads, 0, *(cudaStream_t *)stream_handle->handle>>>(
                input_data, top1_group_data, topk_indice_data, output_data, num_local_gpus, COL
                );
    }else{
        sammax_kernel<<<blocks, threads>>>(
                input_data, top1_group_data, topk_indice_data, output_data, num_local_gpus, COL
                );
    }
    return 0;
}



int DLGpuSamMaxGrad(const DLArrayHandle output_grad, DLArrayHandle input, DLArrayHandle top1_group, DLArrayHandle topk_indice, DLArrayHandle output, int num_local_gpus, DLStreamHandle stream_handle){
    assert(output->ndim == 2); 
    assert(input->ndim == 2); // In MoE we need to find the topk of 2-d tensor on the last dimention
    int ROW = input->shape[0];
    assert(output->shape[0]==ROW);
    int COL = input->shape[1];
    assert(output->shape[1]==COL);
    assert(top1_group->ndim==2);
    assert(top1_group->shape[0]==ROW);
    assert(top1_group->shape[1]==1);
    assert(topk_indice->ndim==2);
    assert(topk_indice->shape[0]==ROW);
    assert(topk_indice->shape[1]==1);
    assert(output_grad->ndim==2);
    assert(output_grad->shape[0]==ROW);
    assert(output_grad->shape[1]==COL);

    float* output_grad_data = (float*)output_grad->data;
    float* input_data = (float*) input->data;
    float* top1_group_data = (float*)top1_group->data;
    float* topk_indice_data = (float*)topk_indice->data;
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
        sammax_grad_gate_kernel<<<blocks, threads, 0, *(cudaStream_t *)stream_handle->handle>>>(
                output_grad_data, input_data, top1_group_data, topk_indice_data, output_data, num_local_gpus, COL
                );
    }else{
        sammax_grad_gate_kernel<<<blocks, threads>>>(
                output_grad_data, input_data, top1_group_data, topk_indice_data, output_data, num_local_gpus, COL
                );
    }
    return 0;
}

