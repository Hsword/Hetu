#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include "../cuda_common/gpu_runtime.h"
#include "nccl.h"
#define THREADS_PER_BLOCKS 1024

#define CUDACHECK(cmd)                                                         \
    do {                                                                       \
        cudaError_t e = cmd;                                                   \
        if (e != cudaSuccess) {                                                \
            printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__,      \
                   cudaGetErrorString(e));                                     \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define NCCLCHECK(cmd)                                                         \
    do {                                                                       \
        ncclResult_t r = cmd;                                                  \
        if (r != ncclSuccess) {                                                \
            printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__,      \
                   ncclGetErrorString(r));                                     \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

extern "C" {
void create_streams(cudaStream_t *stream, int *devices, int devices_numbers);

void update_stream(size_t dev_id, cudaStream_t *stream,
                   cudaStream_t *stream_handle);

void free_streams(cudaStream_t *stream, int *devices, int devices_numbers);

void init_NCCL(ncclComm_t *comms, int *devices, int devices_numbers);

void finish_NCCL(ncclComm_t *comms, int devices_numbers);

void Synchronize_streams(cudaStream_t *stream, int *devices,
                         int devices_numbers);

void NCCL_AllReduce(float **sendbuff, float **recvbuff, int size,
                    ncclComm_t *comms, cudaStream_t *stream,
                    int devices_numbers);

void display(const float *device_data, int dev_id, int size);

void show_int(int a);

void create(int **a, int n);

void for_each(int *a, int n);
}
