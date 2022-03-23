#include "gpu_runtime.h"

__global__ void encode_forward(const float* input_data, float* indices_s_data, float* location_s_data, float* output_data, int capacity, int samples, int hidden){
    for (int i = blockIdx.x; i < samples; i += gridDim.x){
        if (location_s_data[i] < capacity && indices_s_data[i] >= 0) {
            for (int j = threadIdx.x; j < hidden; j += 1024){
                output_data[((int)indices_s_data[i] * capacity + (int)location_s_data[i]) * (hidden) + j]=input_data[i * (hidden) + j];                   
            }    
        }         
    }
}


__global__ void encode_backward_data(const float* input_data, float* indices_s_data, float* location_s_data, float* output_data, int capacity, int samples, int hidden){
    for (int i = blockIdx.x; i < samples; i += gridDim.x){
        if (location_s_data[i] < capacity && (int)indices_s_data[i] >= 0) {
            for (int j = threadIdx.x; j < hidden; j += 1024){
                output_data[i * hidden + j] = input_data[(((int)indices_s_data[i]) * capacity + (int)location_s_data[i]) * (hidden) + j];                                                 
            }                                    
        }else{
            for (int j = threadIdx.x; j < hidden; j += 1024){
                output_data[i*hidden+j]=0;    
            }                                  
        }         
    }
}


__global__ void forward(const float* input_data, float* indices_s_data, float* location_s_data, float* gates_data, float* output_data, int capacity, int samples, int hidden){
    for (int i = blockIdx.x; i < samples; i += gridDim.x){
        if (location_s_data[i] < capacity && indices_s_data[i] >= 0) {
            for (int j = threadIdx.x; j < hidden; j += 1024){
                atomicAdd(&output_data[((int)indices_s_data[i] * capacity + (int)location_s_data[i]) * (hidden) + j], gates_data[i] * input_data[i * (hidden) + j]);
            }
        }
    }

}

__global__ void backward_data(const float* input_data, float* indices_s_data, float* location_s_data, float* gates_data, float* output_data, int capacity, int samples, int hidden){
    for (int i = blockIdx.x; i < samples; i += gridDim.x){
        if (location_s_data[i] < capacity && (int)indices_s_data[i] >= 0) {    
            for (int j = threadIdx.x; j < hidden; j += 1024){
                output_data[i * hidden + j] = gates_data[i] * input_data[(((int)indices_s_data[i]) * capacity + (int)location_s_data[i]) * (hidden) + j];        
            }
        }else{
            for (int j = threadIdx.x; j < hidden; j += 1024){
                output_data[i*hidden+j]=0;
            }   
        }         
    }
}


__global__ void forward_no_gate(const float* input_data, float* indices_s_data, float* location_s_data, float* output_data, int capacity, int samples, int hidden){
    for (int i = blockIdx.x; i < samples; i += gridDim.x){
        if (location_s_data[i] < capacity && indices_s_data[i] >= 0) {                        
            for (int j = threadIdx.x; j < hidden; j += 1024){                                        
                atomicAdd(&output_data[((int)indices_s_data[i] * capacity + (int)location_s_data[i]) * (hidden) + j], input_data[i * (hidden) + j]);
            }    
        }         
    }
}

__global__ void backward_data_no_gate(const float* input_data, float* indices_s_data, float* location_s_data, float* output_data, int capacity, int samples, int hidden){ 
    for (int i = blockIdx.x; i < samples; i += gridDim.x){            
        if (location_s_data[i] < capacity && (int)indices_s_data[i] >= 0) {                        
            for (int j = threadIdx.x; j < hidden; j += 1024){
                output_data[i * hidden + j] = input_data[(((int)indices_s_data[i]) * capacity + (int)location_s_data[i]) * (hidden) + j];                                                 
            }                                   
        }else{ 
            for (int j = threadIdx.x; j < hidden; j += 1024){
                output_data[i*hidden+j]=0;                                                 
            }                                  
        }         
    }
}



__global__ void backward_gate(const float* dispatched_input_data, float* reshaped_input_data, float* indices_s_data, float* location_s_data, float* output_data, int capacity, int samples, int hidden){
    if(location_s_data[(int)blockIdx.x]>=capacity || indices_s_data[(int)blockIdx.x]<0){
        if((int)threadIdx.x==0){
            output_data[(int)blockIdx.x] = 0;
        }
        return;
    }
    int indice = (int)indices_s_data[(int)blockIdx.x]*capacity + location_s_data[(int)blockIdx.x];

    float grad_gates_rf = 0.0;
    for(int i=threadIdx.x; i<hidden; i+=32){
        grad_gates_rf += dispatched_input_data[indice*(hidden)+i]*reshaped_input_data[(int)blockIdx.x*hidden+i];
    }
    
    float red_buf0[1];
    unsigned int mask[1];
    float t0[1];
    red_buf0[0]=grad_gates_rf;
    mask[0] = __activemask();
    t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 16, 32);
    red_buf0[0] = red_buf0[0]+t0[0];
    t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 8, 32);
    red_buf0[0] = red_buf0[0]+t0[0];
    t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 4, 32);
    red_buf0[0] = red_buf0[0]+t0[0];
    t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 2, 32);
    red_buf0[0] = red_buf0[0]+t0[0];
    t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 1, 32);
    red_buf0[0] = red_buf0[0]+t0[0];
    red_buf0[0] = __shfl_sync(mask[0], red_buf0[0], 0, 32);

    if((int)threadIdx.x==0){
        output_data[(int)blockIdx.x] = red_buf0[0];
    }
    return;
}

int DLGpuLayoutTransformTop1(const DLArrayHandle input, DLArrayHandle indices_s, DLArrayHandle location_s,
    DLArrayHandle output, int capacity, DLStreamHandle stream_handle){
   

    assert(input->ndim == 2); // (num_sample * model_dim)
    int samples = input->shape[0];
    int model_dim = input->shape[1];

    assert(indices_s -> ndim == 2); // (num_sample, 1)
    assert(indices_s->shape[1]==1);

    assert(location_s->ndim == 1); // (num_sample)
    assert(output->ndim == 2); // same as indices_s


    const float* input_data = (const float*)input->data;
    float* indices_s_data = (float*)indices_s->data;
    float* location_s_data = (float*)location_s->data;
    float* output_data = (float*)output->data;

    dim3 blocks;
    dim3 threads;

    threads.x = 1024;
    blocks.x = 128;



    if(stream_handle){
        encode_forward<<<blocks, threads, 0, *(cudaStream_t *)stream_handle->handle>>>(
                input_data, indices_s_data, location_s_data, output_data, capacity, samples, model_dim);
    }else{
        encode_forward<<<blocks, threads>>>(
                input_data, indices_s_data, location_s_data, output_data, capacity, samples, model_dim);
    }
    return 0;
}


int DLGpuLayoutTransformTop2(const DLArrayHandle input, DLArrayHandle indices_s1, DLArrayHandle indices_s2, DLArrayHandle location_s1, DLArrayHandle location_s2, DLArrayHandle output, int capacity, DLStreamHandle stream_handle){
    
    assert(input->ndim == 2); // (num_sample * model_dim)
    int samples = input->shape[0];
    int model_dim = input->shape[1];
    assert(indices_s1 -> ndim == 2); // (num_sample)
    assert(indices_s2 -> ndim == 2); 
    assert(location_s1 -> ndim == 1); // (num_sample)
    assert(location_s2 -> ndim == 1);                    
    assert(output->ndim == 2); // same as indices_s

    const float* input_data = (const float*)input->data;
    float* indices_s1_data = (float*)indices_s1->data;
    float* indices_s2_data = (float*)indices_s2->data;                 
    float* location_s1_data = (float*)location_s1->data;
    float* location_s2_data = (float*)location_s2->data;                        
    float* output_data = (float*)output->data;
                                        
    dim3 blocks;
    dim3 threads;
    threads.x = 1024;
    blocks.x = 128;
    
    if(stream_handle){
        encode_forward<<<blocks, threads, 0, *(cudaStream_t *)stream_handle->handle>>>(input_data, indices_s1_data, location_s1_data, output_data, capacity, samples, model_dim);
        encode_forward<<<blocks, threads, 0, *(cudaStream_t *)stream_handle->handle>>>(input_data, indices_s2_data, location_s2_data, output_data, capacity, samples, model_dim);
    }else{
        encode_forward<<<blocks, threads>>>(input_data, indices_s1_data, location_s1_data, output_data, capacity, samples, model_dim);
        encode_forward<<<blocks, threads>>>(input_data, indices_s2_data, location_s2_data, output_data, capacity, samples, model_dim);
    }
    
    return 0;
}

int DLGpuReverseLayoutTransformTop1(const DLArrayHandle input, DLArrayHandle indices_s, DLArrayHandle location_s, DLArrayHandle gates, \
            DLArrayHandle output, int capacity, DLStreamHandle stream_handle){
    assert(input->ndim == 2);
    assert(output->ndim == 2);
    int samples = output->shape[0];
    int model_dim = output->shape[1]; 
    assert(indices_s->ndim == 2);
    assert(indices_s->shape[1] == 1);
    assert(location_s->ndim == 1);
    assert(gates->ndim==1);
    assert(output->ndim==2);
    const float* input_data = (const float*)input->data;
    float* indices_s_data = (float*)indices_s->data;
    float* location_s_data = (float*)location_s->data;
    float* gates_data = (float*)gates->data;
    float* output_data = (float*)output->data;
    dim3 blocks;
    dim3 threads;
    blocks.x = 128;
    threads.x = 1024;
    if(stream_handle){
        backward_data<<<blocks, threads, 0, *(cudaStream_t *)stream_handle->handle>>>(input_data, indices_s_data, location_s_data, gates_data, output_data, capacity, samples, model_dim);                                   
    }else{ 
        backward_data<<<blocks, threads>>>(input_data, indices_s_data, location_s_data, gates_data, output_data, capacity, samples, model_dim);      
    }
    return 0;
}

int DLGpuReverseLayoutTransformTop2(const DLArrayHandle input, DLArrayHandle indices_s1, DLArrayHandle indices_s2, DLArrayHandle location_s1, \
            DLArrayHandle location_s2, DLArrayHandle gates_1, DLArrayHandle gates_2, DLArrayHandle output, int capacity, DLStreamHandle stream_handle){
    assert(input->ndim == 2);
    assert(output->ndim == 2);
    int samples = output->shape[0];
    int model_dim = output->shape[1];
    assert(indices_s1->ndim == 2);
    assert(indices_s2->ndim == 2);
    assert(indices_s1->shape[1] == 1);
    assert(indices_s2->shape[1] == 1);
    assert(location_s1->ndim == 1);
    assert(location_s1->ndim == 1);
    assert(gates_1->ndim==1);      
    assert(gates_2->ndim==1);
    assert(output->ndim==2);
    const float* input_data = (const float*)input->data;
    float* indices_s1_data = (float*)indices_s1->data;
    float* indices_s2_data = (float*)indices_s2->data;
    float* location_s1_data = (float*)location_s1->data;
    float* location_s2_data = (float*)location_s2->data;
    float* gates_1_data = (float*)gates_1->data;
    float* gates_2_data = (float*)gates_2->data;
    float* output_data = (float*)output->data;
    dim3 blocks;
    dim3 threads;
    blocks.x = 128;                                                                     
    threads.x = 1024;                     
    if(stream_handle){                                                                          backward_data<<<blocks, threads, 0, *(cudaStream_t *)stream_handle->handle>>>(input_data, indices_s1_data, location_s1_data, gates_1_data, output_data, capacity, samples, model_dim);
        backward_data<<<blocks, threads, 0, *(cudaStream_t *)stream_handle->handle>>>(input_data, indices_s2_data, location_s2_data, gates_2_data, output_data, capacity, samples, model_dim);
    }else{  

        backward_data<<<blocks, threads>>>(input_data, indices_s1_data, location_s1_data, gates_1_data, output_data, capacity, samples, model_dim);  
        backward_data<<<blocks, threads>>>(input_data, indices_s2_data, location_s2_data, gates_2_data, output_data, capacity, samples, model_dim);    
    }
    return 0;
}

int DLGpuLayoutTransformTop1Gradient(const DLArrayHandle input, DLArrayHandle indice, DLArrayHandle location, DLArrayHandle output, int capacity, DLStreamHandle stream_handle){
    assert(input->ndim == 2);
    assert(output->ndim == 2);
    int samples = output->shape[0];
    int model_dim  = output->shape[1];
    assert(indice->ndim==2);
    assert(indice->shape[1]==1);
    assert(location->ndim==1);
    float* input_data = (float*)input->data;
    float* indice_data = (float*)indice->data;
    float* location_data = (float*)location->data;
    float* output_data = (float*)output->data;
    dim3 blocks;
    dim3 threads;
    blocks.x = 128;
    threads.x = 1024;
    if(stream_handle){
        encode_backward_data<<<blocks, threads, 0, *(cudaStream_t *)stream_handle->handle>>>(input_data, indice_data, location_data, output_data, capacity, samples, model_dim);
    }else{
        encode_backward_data<<<blocks, threads>>>(input_data, indice_data, location_data, output_data, capacity, samples, model_dim);
    }
    return 0;
}


int DLGpuReverseLayoutTransformTop1GradientData(const DLArrayHandle input, DLArrayHandle indice, DLArrayHandle location, DLArrayHandle gate, DLArrayHandle output, int capacity, DLStreamHandle stream_handle){
    assert(input->ndim==2);
    assert(output->ndim==2);
    int samples = input->shape[0];
    int model_dim =  input->shape[1];
    assert(indice->ndim==2);
    assert(indice->shape[1]==1);
    assert(location->ndim==1);
    assert(gate->ndim==1);

    float* input_data = (float*)input->data;
    float* output_data = (float*)output->data;
    float* indice_data = (float*)indice->data;
    float* location_data = (float*)location->data;
    float* gate_data = (float*)gate->data;

    dim3 threads;
    dim3 blocks;
    
    blocks.x = 128;
    threads.x = 1024;

    if(stream_handle){
        forward<<<blocks, threads, 0, *(cudaStream_t *)stream_handle->handle>>>(input_data, indice_data, location_data, gate_data, output_data, capacity, samples, model_dim);
    }else{
        forward<<<blocks, threads>>>(input_data, indice_data, location_data, gate_data, output_data, capacity, samples, model_dim);
    }
    return 0;
}


int DLGpuReverseLayoutTransformTop2GradientData(const DLArrayHandle input, DLArrayHandle indice_1, DLArrayHandle indice_2, DLArrayHandle location_1, DLArrayHandle location_2, DLArrayHandle gate_1, DLArrayHandle gate_2, DLArrayHandle output, int capacity, DLStreamHandle stream_handle){
    assert(input->ndim==2);
    assert(output->ndim==2);
    int samples = input->shape[0];
    int model_dim = input->shape[1];
    assert(indice_1->ndim==2);
    assert(indice_2->ndim==2);
    assert(indice_1->shape[1]==1);
    assert(indice_2->shape[1]==1);
    assert(location_1->ndim==1);
    assert(location_2->ndim==1);
    assert(gate_1->ndim==1);
    assert(gate_2->ndim==1);

    float* input_data = (float*)input->data;
    float* output_data = (float*)output->data;
    float* indice_1_data = (float*)indice_1->data;
    float* indice_2_data = (float*)indice_2->data;
    float* location_1_data = (float*)location_1->data;
    float* location_2_data = (float*)location_2->data;
    float* gate_1_data = (float*)gate_1->data;
    float* gate_2_data = (float*)gate_2->data;

    dim3 blocks;
    dim3 threads;
    blocks.x = 128;
    threads.x = 1024;

    if(stream_handle){
        forward<<<blocks, threads, 0, *(cudaStream_t *)stream_handle->handle>>>(input_data, indice_1_data, location_1_data, gate_1_data, output_data, capacity, samples, model_dim);
        forward<<<blocks, threads, 0, *(cudaStream_t *)stream_handle->handle>>>(input_data, indice_2_data, location_2_data, gate_2_data, output_data, capacity, samples, model_dim);
    }else{
        forward<<<blocks, threads>>>(input_data, indice_1_data, location_1_data, gate_1_data, output_data, capacity, samples, model_dim);
        forward<<<blocks, threads>>>(input_data, indice_1_data, location_1_data, gate_1_data, output_data, capacity, samples, model_dim);
    }
    return 0;
}

int DLGpuReverseLayoutTransformTop1GradientGate(const DLArrayHandle combined_output,DLArrayHandle expert_output, DLArrayHandle indice, DLArrayHandle location, DLArrayHandle output, int capacity, DLStreamHandle stream_handle){
    
    assert(combined_output->ndim==2);
    assert(expert_output->ndim==2);
    int samples = combined_output->shape[0];
    int model_dim = expert_output->shape[1];
    assert(indice->ndim == 2);
    assert(indice->shape[1]==1);
    assert(location->ndim == 1);
    assert(output->ndim == 1);

    float* combined_output_data = (float*)combined_output->data;
    const float* expert_output_data = (const float*)expert_output->data;
    float* indice_data = (float*)indice->data;
    float* location_data = (float*)location->data;
    float* output_data = (float*)output->data;

    dim3 threads;
    dim3 blocks;
    blocks.x = samples;
    threads.x = 32;

    if(stream_handle){
        backward_gate<<<blocks, threads, 0, *(cudaStream_t *)stream_handle->handle>>>(expert_output_data, combined_output_data, indice_data, location_data, output_data, capacity, samples, model_dim);
    }else{
        backward_gate<<<blocks, threads>>>(expert_output_data, combined_output_data, indice_data, location_data, output_data, capacity, samples, model_dim);
    }

    return 0;
}


int DLGpuReverseLayoutTransformNoGate(const DLArrayHandle input, DLArrayHandle indices_s, DLArrayHandle location_s, \
                    DLArrayHandle output, int capacity, DLStreamHandle stream_handle){
    assert(input->ndim == 2);       
    assert(output->ndim == 2);            
    int samples = output->shape[0];                
    int model_dim = output->shape[1];
    assert(indices_s->ndim == 2);
    assert(indices_s->shape[1] == 1);                            
    assert(location_s->ndim == 1);                                               
    assert(output->ndim==2);                       
    const float* input_data = (const float*)input->data;                        
    float* indices_s_data = (float*)indices_s->data;                                                
    float* location_s_data = (float*)location_s->data;                                                
    float* output_data = (float*)output->data;                                        
    dim3 blocks;                                          
    dim3 threads;            
    blocks.x = 128;             
    threads.x = 1024;               
    if(stream_handle){             
        backward_data_no_gate<<<blocks, threads, 0, *(cudaStream_t *)stream_handle->handle>>>(input_data, indices_s_data, location_s_data, output_data, capacity, samples, model_dim);                       
    }else{                                                                                                    
        backward_data_no_gate<<<blocks, threads>>>(input_data, indices_s_data, location_s_data, output_data, capacity, samples, model_dim);
    }                                                                                
    return 0;
}

int DLGpuReverseLayoutTransformNoGateGradient(const DLArrayHandle input, DLArrayHandle indice, DLArrayHandle location, DLArrayHandle output, int capacity, DLStreamHandle stream_handle){
    assert(input->ndim==2);
    assert(output->ndim==2);
    int samples = input->shape[0];
    int model_dim =  input->shape[1];
    assert(indice->ndim==2);
    assert(indice->shape[1]==1);                        
    assert(location->ndim==1);                                                  
    float* input_data = (float*)input->data;                                        
    float* output_data = (float*)output->data;                                        
    float* indice_data = (float*)indice->data;
    float* location_data = (float*)location->data;
    dim3 threads;                                      
    dim3 blocks;
    blocks.x = 128;            
    threads.x = 1024;
               
    if(stream_handle){
        forward_no_gate<<<blocks, threads, 0, *(cudaStream_t *)stream_handle->handle>>>(input_data, indice_data, location_data, output_data, capacity, samples, model_dim);                  
    }else{ 
        forward_no_gate<<<blocks, threads>>>(input_data, indice_data, location_data,output_data, capacity, samples, model_dim);      
    }                                                                            
    return 0;
}

