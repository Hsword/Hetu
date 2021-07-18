# compile nccl and mpi communication

add_library(_nccl_runtime SHARED nccl_communication.cu)
add_library(_mpi_nccl_runtime_api SHARED mpi_nccl_communication.cu)
add_library(_mpi_runtime_api SHARED mpi_communication.cc)

find_package(NCCL 2.8 REQUIRED)

find_package(MPI 3.1)
if(NOT MPI_FOUND)
    message(STATUS "MPI with version >= 3.1 not Found, Preparing OpenMPI ...")
    FetchContent_Declare(openmpi URL https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.3.tar.gz)
    message(STATUS "Fetching openmpi source code ...")
    FetchContent_MakeAvailable(openmpi)
    set(MPI_CXX_LIBRARIES ${openmpi_BINARY_DIR}/lib/libmpi.so)
    set(MPI_CXX_INCLUDE_DIRS ${openmpi_BINARY_DIR}/include)
    add_custom_command(OUTPUT ${MPI_CXX_INCLUDE_DIRS}/mpi.h
        COMMAND ./configure --prefix=${openmpi_BINARY_DIR}
        COMMAND make -j8
        COMMAND make install
        COMMENT "configure, compile and install openmpi"
        WORKING_DIRECTORY ${openmpi_SOURCE_DIR}
    )
    add_custom_target(compile_openmpi DEPENDS ${MPI_CXX_INCLUDE_DIRS}/mpi.h)
    add_dependencies(_mpi_runtime_api compile_openmpi)
    add_dependencies(_mpi_nccl_runtime_api compile_openmpi)
endif()

# compile nccl communication
target_link_libraries(_nccl_runtime PUBLIC ${NCCL_LIBRARIES})
target_include_directories(_nccl_runtime PUBLIC
    ${NCCL_INCLUDE_DIRS} ${CUDAToolkit_INCLUDE_DIRS} ${CUDNN_INCLUDE_PATH})
add_custom_target(nccl DEPENDS _nccl_runtime)

# compile nccl+mpi communication
target_link_libraries(_mpi_nccl_runtime_api PUBLIC ${MPI_CXX_LIBRARIES} ${NCCL_LIBRARIES})
target_include_directories(_mpi_nccl_runtime_api PUBLIC
    ${NCCL_INCLUDE_DIRS} ${MPI_CXX_INCLUDE_DIRS} ${CUDAToolkit_INCLUDE_DIRS} ${CUDNN_INCLUDE_PATH})
add_custom_target(mpi_nccl DEPENDS _mpi_nccl_runtime_api)

# compile mpi communication
target_include_directories(_mpi_runtime_api PUBLIC ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(_mpi_runtime_api PUBLIC ${MPI_CXX_LIBRARIES})
add_custom_target(mpi DEPENDS _mpi_runtime_api)
add_custom_target(allreduce)
add_dependencies(allreduce _nccl_runtime _mpi_nccl_runtime_api)
add_dependencies(allreduce _mpi_runtime_api)
