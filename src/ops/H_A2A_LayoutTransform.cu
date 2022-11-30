#include "gpu_runtime.h"

__global__ void ha2a_layout_transform_kernel(const float* input_data, float* output_data, int samples, int hidden, int num_nodes, int num_local_gpus){
    int data_size_per_gpu = samples/num_local_gpus;
    int data_size_per_gpu_per_node = data_size_per_gpu/(num_nodes);
    int data_size_per_gpu_per_gpu = data_size_per_gpu/(num_local_gpus*num_nodes);
    int data_size_per_node = samples/num_nodes;
    int gpu_id = 0;
    int target_node_id = 0;
    int target_gpu_id = 0;
    int tmp = 0;
    int offset=0;
    for (int i = blockIdx.x; i < samples; i += gridDim.x){
        gpu_id = i/data_size_per_gpu;
        tmp = i%data_size_per_gpu;
        target_node_id = tmp/data_size_per_gpu_per_node;
        tmp = tmp%data_size_per_gpu_per_node;
        target_gpu_id = tmp/data_size_per_gpu_per_gpu;
        offset = tmp%data_size_per_gpu_per_gpu;
        for (int j = threadIdx.x; j < hidden; j += 1024){
            output_data[(target_node_id*data_size_per_node+target_gpu_id*data_size_per_gpu_per_node+gpu_id*data_size_per_gpu_per_gpu+offset) * (hidden) + j]=input_data[i * (hidden) + j];                   
        }    
    }
}

__global__ void ha2a_reverse_layout_transform_kernel(const float* input_data, float* output_data, int samples, int hidden, int num_nodes, int num_local_gpus){
    int data_size_per_gpu = samples/num_local_gpus;
    int data_size_per_gpu_per_node = data_size_per_gpu/(num_nodes);
    int data_size_per_gpu_per_gpu = data_size_per_gpu/(num_nodes*num_local_gpus);
    int data_size_per_node = samples/num_nodes;
    int gpu_id = 0;
    int target_node_id = 0;
    int target_gpu_id = 0;
    int tmp = 0;
    int offset=0;
    for (int i = blockIdx.x; i < samples; i += gridDim.x){
        target_node_id = i/data_size_per_node;
        tmp = i%data_size_per_node;
        target_gpu_id = tmp/data_size_per_gpu_per_node;
        tmp = tmp%data_size_per_gpu_per_node;
        gpu_id = tmp/data_size_per_gpu_per_gpu;
        offset = tmp%data_size_per_gpu_per_gpu;
        for (int j = threadIdx.x; j < hidden; j += 1024){
            output_data[(target_gpu_id*data_size_per_gpu+target_node_id*data_size_per_gpu_per_node+gpu_id*data_size_per_gpu_per_gpu+offset) * (hidden) + j]=input_data[i * (hidden) + j];
        }
    }
}

int DLGpuHA2ALayoutTransform(const DLArrayHandle input, DLArrayHandle output, int num_nodes, int num_local_gpus, DLStreamHandle stream_handle){

    assert(input->ndim == 2); // (num_sample * model_dim)
    int samples = input->shape[0];
    int model_dim = input->shape[1];
    assert(output->ndim == 2);

    const float* input_data = (const float*)input->data;
    float* output_data = (float*)output->data;

    dim3 blocks;
    dim3 threads;

    threads.x = 1024;
    blocks.x = 128;

    if(stream_handle){
        ha2a_layout_transform_kernel<<<blocks, threads, 0, *(cudaStream_t *)stream_handle->handle>>>(input_data, output_data, samples, model_dim, num_nodes, num_local_gpus);
    }else{
        ha2a_layout_transform_kernel<<<blocks, threads>>>(input_data, output_data, samples, model_dim, num_nodes, num_local_gpus);
    }
    return 0;
}

int DLGpuHA2AReverseLayoutTransform(const DLArrayHandle input, DLArrayHandle output, int num_nodes, int num_local_gpus,  DLStreamHandle stream_handle){
    assert(input->ndim == 2);
    assert(output->ndim == 2);
    int samples = output->shape[0];
    int model_dim = output->shape[1]; 
    const float* input_data = (const float*)input->data;
    float* output_data = (float*)output->data;
    dim3 blocks;
    dim3 threads;
    blocks.x = 128;
    threads.x = 1024;
    if(stream_handle){
        ha2a_reverse_layout_transform_kernel<<<blocks, threads, 0, *(cudaStream_t *)stream_handle->handle>>>(input_data, output_data, samples, model_dim, num_nodes, num_local_gpus);
    }else{ 
        ha2a_reverse_layout_transform_kernel<<<blocks, threads>>>(input_data, output_data,samples, model_dim, num_nodes, num_local_gpus);      
    }
    return 0;
}

