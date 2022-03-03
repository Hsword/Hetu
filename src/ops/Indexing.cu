#include "gpu_runtime.h"

__global__ void indexing_kernel(const float* input_data, const float* index_data, float* output_data, int col, int row){
    for(int i=blockIdx.x; i<row; i+=gridDim.x){
        for(int j=threadIdx.x; j<col; j+=1024){
            output_data[i*col+j]=input_data[(int)index_data[i]*col + j];
        }
    }
}

__global__ void indexing_grad_kernel(const float* output_grad_data, const float* index_data, float* input_grad_data, int col, int row){
    for(int i=blockIdx.x; i<row; i+=gridDim.x){            
        for(int j=threadIdx.x; j<col; j+=1024){                              
            input_grad_data[(int)index_data[i]*col+j]=output_grad_data[i*col+j];
        }                    
    }
}


int DLGpuIndexing(const DLArrayHandle input, DLArrayHandle index, DLArrayHandle output, DLStreamHandle stream_handle){
    assert(input->ndim == 2);
    assert(output->ndim == 2);
    assert(index->ndim == 1);
    int ROW = input->shape[0];
    int COL = input->shape[1];
    
    const float* input_data = (const float*)input->data;
    float* index_data = (float*)index->data;
    float* output_data = (float*)output->data;

    dim3 blocks;
    dim3 threads;

    blocks.x = 128;
    threads.x = 1024;

    if(stream_handle){
        indexing_kernel<<<blocks, threads, 0, *(cudaStream_t *)stream_handle->handle>>>(
input_data, index_data, output_data, COL, ROW);
    }else{
        indexing_kernel<<<blocks, threads>>>(input_data, index_data, output_data, COL, ROW);
    }
    
    return 0;
}

int DLGpuIndexingGrad(const DLArrayHandle output_grad, DLArrayHandle index, DLArrayHandle input_grad, DLStreamHandle stream_handle){
        assert(output_grad->ndim == 2);
        assert(input_grad->ndim == 2);
        assert(index->ndim == 1);
        int ROW = output_grad->shape[0];
        int COL = input_grad->shape[1];
        const float* output_grad_data = (const float*)output_grad->data;
        float* index_data = (float*)index->data;
        float* input_grad_data = (float*)input_grad->data;
        dim3 blocks;
        dim3 threads;
        blocks.x = 128;
        threads.x = 1024;
        if(stream_handle){
            indexing_kernel<<<blocks, threads, 0, *(cudaStream_t *)stream_handle->handle>>>(
                    output_grad_data, index_data, input_grad_data, COL, ROW);                                      
        }else{
            indexing_kernel<<<blocks, threads>>>(output_grad_data, index_data, input_grad_data, COL, ROW);    }
        return 0;
}

