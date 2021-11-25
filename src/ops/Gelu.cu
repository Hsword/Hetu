#include "gpu_runtime.h"
#define pi 3.14159265358979323846
#define e  2.71828182845904523536

__global__ void Gelu_kernel(float *input, float *output, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    output[ind] = input[ind] * 0.5 * (1.0 + erf( input[ind]/sqrt(2.0)));
    //output[ind] = 0.5 * input[ind] * (1 + tanh(sqrt(2 / pi) * (input[ind] + 0.044715 * pow(input[ind], 3))));
    //printf("output=%f\n",output[ind]);
    //printf("input=%f,output=%f\n",input[ind],output[ind]);
}



int DLGpuGelu(const DLArrayHandle input, DLArrayHandle output,
              DLStreamHandle stream_handle = NULL) {
    size_t size = 1;
    for (index_t i = 0; i < input->ndim; i++) {
        size *= input->shape[i];
    }
    //printf("111111111111\n");
    dim3 blocks;
    dim3 threads;
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;
    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }
    if (stream_handle)
        Gelu_kernel<<<blocks, threads, 0,
                      *(cudaStream_t *)stream_handle->handle>>>(
            input_data, output_data, size);
    else
        Gelu_kernel<<<blocks, threads>>>(input_data, output_data, size);
    return 0;
}

// __global__ void relu_grad_kernel(const float *input, const float *in_grad,
//                                  float *output, size_t size) {
//     size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
//     if (ind >= size)
//         return;
//     float s = 0;
//     if (input[ind] > 0)
//         s = 1;
//     if (input[ind] < 0)
//         s = -1;
//     output[ind] = (s + 1) * in_grad[ind] * 0.5;
// }

__global__ void gelu_grad_kernel(const float *input, const float *in_grad,
                                 float *output, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size)
        return;
    //printf("Grad__\n");
   // output[ind] = in_grad[ind]*(0.5*tanhf(0.0356774*pow(input[ind],3)+0.797885*input[ind])+(0.0535161*pow(input[ind],3)+0.398942*input[ind])*pow(1/(cosh(0.0356774*pow(input[ind],3)+0.797885*input[ind])),2)+0.5);    
    output[ind] = in_grad[ind]*(0.5+0.5*erf( input[ind]/sqrt(2.0))+0.5*input[ind]*(sqrt(2.0)*pow(e,(-0.5*pow(input[ind],2)))/sqrt(pi)));    //printf("input=%f,output=%f\n",input[ind], output[ind]);
}


int DLGpuGeluGradient(const DLArrayHandle input, const DLArrayHandle in_grad,
                      DLArrayHandle output,
                      DLStreamHandle stream_handle = NULL) {
    size_t size = 1;
    for (index_t i = 0; i < input->ndim; i++) {
        size *= input->shape[i];
    }
    // printf("grad\n");
    dim3 blocks;
    dim3 threads;
    const float *input_data = (const float *)input->data;
    const float *in_grad_data = (const float *)in_grad->data;
    float *output_data = (float *)output->data;
    if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
    }
    if (stream_handle)
        gelu_grad_kernel<<<blocks, threads, 0,
                           *(cudaStream_t *)stream_handle->handle>>>(
            input_data, in_grad_data, output_data, size);
    else
        gelu_grad_kernel<<<blocks, threads>>>(input_data, in_grad_data,
                                              output_data, size);
    return 0;
}

//liang 
