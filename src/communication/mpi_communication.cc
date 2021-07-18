#include "../header/mpi_communication.h"

static const MPI_Datatype TYPE2TYPE_V1[] = {
    MPI_CHAR,  MPI_INT,   MPI_UNSIGNED, MPI_LONG_LONG, MPI_UNSIGNED_LONG_LONG,
    MPI_FLOAT, MPI_DOUBLE};

MPI_Datatype _get_proper_Datatype(int datatype) {
    return TYPE2TYPE_V1[datatype];
}

static const MPI_Op TYPE2TYPE_V2[] = {
    MPI_OP_NULL, MPI_MAX,    MPI_MIN,    MPI_SUM,    MPI_PROD,
    MPI_LAND,    MPI_BAND,   MPI_LOR,    MPI_BOR,    MPI_LXOR,
    MPI_BXOR,    MPI_MINLOC, MPI_MAXLOC, MPI_REPLACE};

MPI_Op _get_proper_MPIOp(int optype) {
    return TYPE2TYPE_V2[optype];
}

void MPIInit() {
    // MPICHECK(MPI_Init(argc, &argv));
    MPICHECK(MPI_Init(NULL, NULL));
}

void MPIGetComm(MPI_Comm *comm) {
    *comm = MPI_COMM_WORLD;
}

void getMPICommRank(MPI_Comm *comm, int *myRank) {
    MPICHECK(MPI_Comm_rank(*comm, myRank));
}

void getMPICommSize(MPI_Comm *comm, int *nRanks) {
    MPICHECK(MPI_Comm_size(*comm, nRanks));
}

void dlarrayAllReduce(DLArray *array, int datatype, int op, MPI_Comm *comm) {
    int size = 1;
    for (int i = 0; i < array->ndim; i++) {
        size = size * array->shape[i];
    }
    float *data_buffer = (float *)(array->data);
    MPI_Datatype red_datatype = _get_proper_Datatype(datatype);
    MPI_Op red_Op = _get_proper_MPIOp(op);
    MPICHECK(MPI_Allreduce(MPI_IN_PLACE, data_buffer, size, red_datatype,
                           red_Op, *comm));
}

void MPIFinalize() {
    MPICHECK(MPI_Finalize());
}