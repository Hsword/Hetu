#include "gpu_runtime.h"

int DLGpuClone(const DLArrayHandle input, DLArrayHandle output, DLStreamHandle stream_handle=NULL){
    float* input_data=(float*)input->data;
    float* output_data=(float*)output->data;
    int size = 1;
    for(int i=0;i<input->ndim; i++){
        size*=input->shape[i];
    }
    cudaMemcpy((void*)output_data, (void*)input_data, size*sizeof(float),cudaMemcpyDeviceToDevice);
    return 0;

}

