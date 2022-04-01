#include "gpu_runtime.h"

__global__ void scatter1d_kernel(const float* input_data, float* index_data, float* output_data, int col){
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i<col){
        output_data[(int)index_data[i]]=input_data[i];
    }
}


__global__ void scatter1d_grad_kernel(const float* output_grad_data, float* index_data, float* input_grad_data, int col){
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i<col){
        input_grad_data[i]=output_grad_data[(int)index_data[i]];
    }
}

int DLGpuScatter1D(const DLArrayHandle input, DLArrayHandle index, DLArrayHandle output, DLStreamHandle stream_handle){
    assert(input->ndim == 1);
    assert(output->ndim == 1);
    assert(index->ndim == 1);
    int COL = input->shape[0];
    
    const float* input_data = (const float*)input->data;
    float* index_data = (float*)index->data;
    float* output_data = (float*)output->data;

    dim3 blocks;
    dim3 threads;

    if(COL <= 1024){
        blocks.x = 1;
        threads.x = COL;
    }else{
        blocks.x = (COL+1023)/1024;
        threads.x = 1024;
    }


    if(stream_handle){
        scatter1d_kernel<<<blocks, threads, 0, *(cudaStream_t *)stream_handle->handle>>>(
input_data, index_data, output_data, COL);
    }else{
        scatter1d_kernel<<<blocks, threads>>>(input_data, index_data, output_data, COL);
    }
    
    return 0;
}

int DLGpuScatter1DGrad(const DLArrayHandle output_grad, DLArrayHandle index, DLArrayHandle input_grad, DLStreamHandle stream_handle){
    assert(output_grad->ndim == 1);        
    assert(input_grad->ndim == 1);            
    assert(index->ndim == 1);                
    int COL = input_grad->shape[0];                    
    const float* output_grad_data = (const float*)output_grad->data;
    float* index_data = (float*)index->data;
    float* input_grad_data = (float*)input_grad->data;
    dim3 blocks;
    dim3 threads;
    if(COL <= 1024){         
        blocks.x = 1;            
        threads.x = COL;                 
    }else{                
        blocks.x = (COL+1023)/1024;  
        threads.x = 1024;                               
    }
    if(stream_handle){
        scatter1d_grad_kernel<<<blocks, threads, 0, *(cudaStream_t *)stream_handle->handle>>>(      
                output_grad_data, index_data, input_grad_data, COL);                                                   }else{                                                               
            scatter1d_kernel<<<blocks, threads>>>(output_grad_data, index_data, input_grad_data, COL);
        }                                              
    return 0;
}

