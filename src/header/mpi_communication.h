#include <stdio.h>
#include "mpi.h"
#include <stdlib.h>
#include <unistd.h>
#include "../common/dlarray.h"

#define MPICHECK(cmd)                                                          \
    do {                                                                       \
        int e = cmd;                                                           \
        if (e != MPI_SUCCESS) {                                                \
            printf("Failed: MPI error %s:%d '%d'\n", __FILE__, __LINE__, e);   \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

extern "C" {
void MPIInit();
void MPIGetComm(MPI_Comm *comm);
void getMPICommRank(MPI_Comm *comm, int *myRank);
void getMPICommSize(MPI_Comm *comm, int *nRanks);
void dlarrayAllReduce(DLArray *array, int datatype, int op, MPI_Comm *comm);
void MPIFinalize();
}