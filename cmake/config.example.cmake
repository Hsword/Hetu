######################
### Set targets ######
######################

# hetu main version, choose from (mkl, gpu, all)
# if using mkl (for CPU) or all, OpenMP(*), mkl required
# if using gpu or all, OpenMP(*), CUDA(*), CUDNN(*) required
set(HETU_VERSION "all")

# whether to compile allreduce module
# nccl(*), openmpi required
set(HETU_ALLREDUCE ON)

# whether to compile ps module
# protobuf(*), zeromq required
set(HETU_PS ON)

# whether to compile geometric module (for GNNs)
# pybind11(*), metis(*) required
set(HETU_GEOMETRIC OFF)

# whether to compile cache module (for PS)
# to enable this, you must turn HETU_PS on
# pybind11(*) required
set(HETU_CACHE ON)

# whether to compile Hetu ML Module
set(HETU_ML OFF)
set(HETU_PARALLEL_ML OFF)

######################
### Set paths ########
######################

# CUDA version >= 10.1
set(CUDAToolkit_ROOT /usr/local/cuda)

# NCCL version >= 2.8
set(NCCL_ROOT $ENV{CONDA_PREFIX})

set(CUDNN_ROOT)

# MPI version >= 3.1 (OpenMPI version >= 4.0.3)
# if valid version not found, we'll download and compile it in time (openmpi-4.0.3)
set(MPI_HOME $ENV{CONDA_PREFIX})

# MKL 1.6.1, MKL_ROOT: root directory of mkl, MKL_BUILD: build directory of mkl
# if not found, we'll download and compile it in time
set(MKL_ROOT $ENV{CONDA_PREFIX})
set(MKL_BUILD $ENV{CONDA_PREFIX})

# ZMQ 4.3.2, ZMQ_ROOT: root directory of zeromq, ZMQ_BUILD: build directory of zeromq
# if not found, we'll download and compile it in time
set(ZMQ_ROOT $ENV{CONDA_PREFIX})
set(ZMQ_BUILD $ENV{CONDA_PREFIX})

# CUB & THRUST
set(CUB_ROOT $ENV{CONDA_PREFIX})
set(THRUST_ROOT $ENV{CONDA_PREFIX})
