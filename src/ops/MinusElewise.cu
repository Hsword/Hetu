#include "gpu_runtime.h"

__global__ void ele_minus_kernel(float* input1, float* input2, float* output, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx>=size){
        return;
    }
    output[idx]=input1[idx]-input2[idx];
}

int DLGpuMinusElewise(const DLArrayHandle input1, DLArrayHandle input2, DLArrayHandle output, DLStreamHandle stream_handle=NULL){
    int size=1;
    for(int i=0;i<input1->ndim;i++){
        size*=input1->shape[i];
    }
    dim3 blocks;
    dim3 threads;

    if(size<=1024){
        blocks.x=1;
        threads.x = size;
    }else{
        threads.x=1024;
        blocks.x=(size+1023)/1024;
    }
    
    float* input1_data=(float*)input1->data;
    float* input2_data=(float*)input2->data;
    float* output_data=(float*)output->data;

    if(stream_handle){
        ele_minus_kernel<<<blocks, threads, 0, *(cudaStream_t *)stream_handle->handle>>>(input1_data, input2_data, output_data, size);
    }else{
        ele_minus_kernel<<<blocks, threads>>>(input1_data, input2_data, output_data, size);
    }
    return 0;
}
