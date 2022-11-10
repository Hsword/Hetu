#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "../cuda_common/gpu_runtime.h"
#include "nccl.h"
#include "mpi.h"
#include "../common/dlarray.h"
#define THREADS_PER_BLOCKS 1024

#define MPICHECK(cmd)                                                          \
    do {                                                                       \
        int e = cmd;                                                           \
        if (e != MPI_SUCCESS) {                                                \
            printf("Failed: MPI error %s:%d '%d'\n", __FILE__, __LINE__, e);   \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

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
void MPIInit();
void MPIFinalize();
void MPIGetComm(MPI_Comm *comm);
void MPIBcast(void *buffer, int size, int root, MPI_Comm comm);
void getMPICommRank(MPI_Comm *comm, int *myRank);
void getMPICommSize(MPI_Comm *comm, int *nRanks);
void getLocalRank(MPI_Comm *comm, int nRanks, int myRank, int *localRank,
                  unsigned long long hostHashs[]);
void getGlobalDevice(MPI_Comm *comm, int nRanks, int myRank, int device_id,
                     int hostDevices[]);
void getNcclUniqueId(ncclUniqueId *Id, MPI_Comm mpi_comm, int localRank,
                     int senderRank = 0);
void getGroupNcclUniqueId(ncclUniqueId *Id, MPI_Comm mpi_comm, int rank,
                          int dests[], int group_size, int group_id);
void initNcclCommRank(ncclComm_t *comm, int nranks, ncclUniqueId *commId,
                      int rank, int localRank);
void GroupStart();
void GroupEnd();
void dlarrayAllReduce(DLArray *input_array, DLArray *output_array, int datatype,
                      int op, ncclComm_t comm, DLStreamHandle stream_handle);
void dlarrayReduce(DLArray *input_array, DLArray *output_array, int datatype,
                   int op, int root, ncclComm_t comm,
                   DLStreamHandle stream_handle);
void dlarrayBroadcast(DLArray *input_array, DLArray *output_array, int datatype,
                      int root, ncclComm_t comm, DLStreamHandle stream_handle);
void dlarrayAllGather(DLArray *array, DLArray *output_array, int datatype,
                      ncclComm_t comm, DLStreamHandle stream_handle);
void dlarrayReduceScatter(DLArray *array, DLArray *output_array, int datatype,
                          int op, ncclComm_t comm,
                          DLStreamHandle stream_handle);
void dlarraySend(DLArray *array, int datatype, int target, ncclComm_t comm,
                 DLStreamHandle stream_handle);
void dlarrayRecv(DLArray *array, int datatype, int src, ncclComm_t comm,
                 DLStreamHandle stream_handle);
void dlarrayAllToAll(DLArray *sendarray, DLArray *recvarray, int datatype, \
                     ncclComm_t comm, DLStreamHandle stream_handle, int num_of_peers);
void dlarrayHAllToAll(DLArray *sendarray, DLArray *recvarray, int datatype, ncclComm_t comm, DLStreamHandle stream_handle, int num_nodes, int num_local_gpus);

void dlarrayHA2AGather(DLArray *sendarr, DLArray *recvarr, int datatype, int myrank, int num_local_gpus, ncclComm_t comm, DLStreamHandle stream_handle);

void dlarrayHA2AScatter(DLArray *sendarr, DLArray *recvarr, int datatype, int myrank, int num_local_gpus, ncclComm_t comm, DLStreamHandle stream_handle);

void commDestroyNccl(ncclComm_t *comm);
void setDevice(int device_id);
}
