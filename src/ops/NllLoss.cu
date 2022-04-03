#include "gpu_runtime.h"

__global__ void nll_loss_kernel(const float* input_data, float* target_data, float* output_data, int rows, int cols){
    int tid = threadIdx.x;
    float partialSum = 0;
    for(int i=blockIdx.x*blockDim.x+tid; i<rows; i+=blockDim.x*gridDim.x){
        partialSum -= input_data[i*cols+(int)target_data[i]];
    }
    atomicAdd(output_data, partialSum/rows);
}

__global__ void nll_loss_grad(const float* output_grad_data, float* target_data, float* input_grad_data, int rows, int cols){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<rows){
        input_grad_data[i*cols+(int)target_data[i]]=-1*output_grad_data[0]/rows;
    }
}

int DLGpuNllLoss(const DLArrayHandle input, DLArrayHandle target, DLArrayHandle output, DLStreamHandle stream_handle){
    assert(input->ndim==2); 
    assert(output->ndim==1);
    assert(output->shape[0]==1);
    int rows = input->shape[0];
    int cols = input->shape[1];
    const float* input_data = (const float*)input->data;
    float* target_data = (float*)target->data;
    float* output_data = (float*)output->data;
    int blocks = 128;
    int threads = 512;
    if(stream_handle){
        nll_loss_kernel<<<blocks, threads, 0, *(cudaStream_t *)stream_handle->handle>>>(
                input_data, target_data, output_data, rows, cols
                );
    }else{
        nll_loss_kernel<<<blocks, threads>>>(input_data, target_data, output_data, rows, cols);
    }
    return 0;
}

int DLGpuNllLossGrad(const DLArrayHandle output_grad, DLArrayHandle target, DLArrayHandle input_grad, DLStreamHandle stream_handle){
    assert(output_grad->ndim==1);
    assert(output_grad->shape[0]==1);
    assert(input_grad->ndim==2);
    int rows = input_grad->shape[0];
    int cols = input_grad->shape[1];
    const float* output_grad_data = (const float*)output_grad->data;
    float* target_data = (float*)target->data;
    float* input_grad_data = (float*)input_grad->data;
    
    dim3 blocks;
    dim3 threads;
    if(rows<=1024){
        blocks.x = 1;
        threads.x = rows;
    }else{
        threads.x = 1024;
        blocks.x = (rows+1023)/1024;
    }
    
    if(stream_handle){
        nll_loss_grad<<<blocks, threads, 0, *(cudaStream_t *)stream_handle->handle>>>(output_grad_data, target_data, input_grad_data, rows, cols);
    }else{
        nll_loss_grad<<<blocks, threads>>>(output_grad_data, target_data, input_grad_data, rows, cols);
    }
    return 0;
}
