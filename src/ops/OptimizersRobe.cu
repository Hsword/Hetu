#include "gpu_runtime.h"


__global__ void sgd_robe_update(const float *grad_data,
                                  const int *indices_data, const int *x_data, float *param_data,
                                  size_t size, size_t length, float lr, int roarsz,
                                  int Bg, int Cg, int Dg,int Z,int blk,int MO) {
    size_t thread_ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_ind >= size)
        return;
    size_t index = thread_ind / length;
    size_t i = thread_ind % length;
    const float cur_grad = grad_data[thread_ind];

    const int X = x_data[index];
    int sgn = (((1ll * X * Bg + 1ll * i * Cg + Dg)%MO+MO)%MO %2)*2-1;
    
    
    int id = indices_data[index * blk+i/Z];
    if (id < 0)
        return;
    atomicAdd(param_data + ((id+i%Z)<roarsz?(id+i%Z):(id+i%Z)-roarsz), -lr * cur_grad * sgn);
//    param_data[(id + offset<roarsz?id+offset:id+offset-roarsz)] -= lr * cur_grad;
}

int SGDOptimizerRobeUpdate(DLArrayHandle param,
                             const DLArrayHandle grad_indices,
                             const DLArrayHandle grad_values,
                             const DLArrayHandle grad_x,
                             float lr, int Bg, int Cg, int Dg, int Z, int MO,
                             DLStreamHandle stream_handle = NULL) {
    size_t size = ArrSize(grad_values);
    size_t roarsz = ArrSize(param);
    int blk = grad_indices->shape[(grad_indices->ndim) - 1];
    size_t length = (grad_values->shape[(grad_values->ndim) - 1]);

    const float *grad_data = (const float *)grad_values->data;
    float *param_data = (float *)param->data;
    const int *indices_data = (const int *)grad_indices->data;
    const int *x_data = (const int *)grad_x->data;
/*
    printf("%d\n",roarsz);

    printf("%d\n",grad_indices->ndim);
    for (int i=0;i<(grad_indices->ndim);++i)
        printf("%d ",(grad_indices->shape[i]));
    printf("\n");

    printf("%d\n",grad_values->ndim);
    for (int i=0;i<(grad_values->ndim);++i)
        printf("%d ",(grad_values->shape[i]));
    printf("\n");

    printf("%d\n",grad_x->ndim);
    for (int i=0;i<(grad_x->ndim);++i)
        printf("%d ",(grad_x->shape[i]));
    printf("\n");
*/

    dim3 blocks;
    dim3 threads;
    ThreadBlock1D(threads, blocks, size);


    if (stream_handle ){
        sgd_robe_update<<<blocks, threads, 0,
                            *(cudaStream_t *)stream_handle->handle>>>(
            grad_data, indices_data, x_data, param_data, size, length, lr, roarsz, Bg, Cg, Dg,Z,blk,MO);
    }
    else{
        sgd_robe_update<<<blocks, threads>>>(grad_data, indices_data, x_data,
                                               param_data, size, length, lr, roarsz, Bg, Cg, Dg,Z,blk,MO);
    }
    
    return 0;
}
