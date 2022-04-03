#include "gpu_runtime.h"

__global__ void scatter_kernel(float* target_data, float* index_data, float* src_data, int tgt_col, int src_col, int row){
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    if(offset >= row){
        return ;
    }
    float* target_data_start = target_data + offset*tgt_col;
    float* index_data_start = index_data + offset*src_col;
    float* src_data_start = src_data + offset*src_col;

    for(int i=0; i<src_col; i++){
        target_data_start[int(index_data_start[i])]=src_data_start[i];
    }
}

int DLGpuScatter(const DLArrayHandle target, int dim, DLArrayHandle index, DLArrayHandle src, DLStreamHandle stream_handle = NULL){
    assert(target->ndim == 2);
    assert(src->ndim == 2);
    assert(index->ndim == 2);
    
    int ROW = target->shape[0];
    int COL = target->shape[1];
    int SRC_COL = src->shape[1];

    float* target_data = (float*) target->data;
    float* src_data = (float*) src->data;
    float* index_data = (float*) index->data;

    dim3 blocks;
    dim3 threads;

    assert(dim == 1);
    // dim = 0 not implemented yet
    // will implement it later

    if(dim == 1){
        if(ROW<=1024){
            blocks.x = 1;
            threads.x = ROW;
        }else{
            blocks.x = (ROW+1023)/1024;
            threads.x = 1024;
        }
        if(stream_handle){
            scatter_kernel<<<blocks, threads, 0, *(cudaStream_t *)stream_handle->handle>>>(target_data, index_data, src_data, COL, SRC_COL, ROW);
        }else{
            scatter_kernel<<<blocks, threads>>>(target_data, index_data, src_data, COL, SRC_COL, ROW);
        }
    }
    return 0;
}
