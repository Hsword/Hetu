cmake_minimum_required(VERSION 3.18)

project(Hetu CXX)

include(cmake/config.cmake)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_CXX_FLAGS "-O3 -Wall")

# openmp
find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# compile flag
if(${HETU_VERSION} STREQUAL "all")
    set(HETU_COMPILE_GPU ON)
    set(HETU_COMPILE_MKL ON)
elseif(${HETU_VERSION} STREQUAL "gpu")
    set(HETU_COMPILE_GPU ON)
    set(HETU_COMPILE_MKL OFF)
elseif(${HETU_VERSION} STREQUAL "mkl")
    set(HETU_COMPILE_GPU OFF)
    set(HETU_COMPILE_MKL ON)
else()
    message(FATAL_ERROR "unknown hetu version")
endif()
message(STATUS "HETU version: ${HETU_VERSION}")

# cuda
if(${HETU_COMPILE_GPU})
    set(CMAKE_CUDA_COMPILER ${CUDAToolkit_ROOT}/bin/nvcc)
    if(EXISTS ${CUDAToolkit_ROOT}/version.txt)
        file(READ ${CUDAToolkit_ROOT}/version.txt RAW_CUDA_VERSION)
    else()
        file(READ ${CUDAToolkit_ROOT}/version.json RAW_CUDA_VERSION)
    endif()
    string(REGEX MATCH "[0-9\.]+" CUDA_VERSION ${RAW_CUDA_VERSION})
    if(${CUDA_VERSION} VERSION_LESS "10.1")
        message(FATAL_ERROR "Required CUDA version >= 10.1, while current CUDA version is ${CUDA_VERSION}")
    endif()
    find_package(CUDAToolkit REQUIRED)
    enable_language(CUDA)
    if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_LESS "10.1")
        message(FATAL_ERROR "Required CUDA version >= 10.1, while current CUDA version is ${CMAKE_CUDA_COMPILER_VERSION}")
    endif()
endif()

include(FetchContent) # download third_party

add_subdirectory(${CMAKE_SOURCE_DIR}/src)

if(${HETU_PS})
    add_subdirectory(${CMAKE_SOURCE_DIR}/ps-lite)
endif()

if(${HETU_GEOMETRIC})
    add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/GraphMix)
endif()
if (HETU_ML)
    add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/HetuML)
endif()
enable_testing()
