# port from https://github.com/dmlc/ps-lite/blob/master/CMakeLists.txt

aux_source_directory(src PS_SRC)
add_library(ps SHARED ${PS_SRC})
target_include_directories(ps PUBLIC include)

# find and build zeroMQ
find_package(ZMQ 4.3.2)
if(NOT ZMQ_FOUND)
    message(STATUS "ZMQ not Found, Preparing ZMQ ...")
    FetchContent_Declare(zmq URL https://github.com/zeromq/libzmq/releases/download/v4.3.2/zeromq-4.3.2.zip)
    FetchContent_GetProperties(zmq)
    if(NOT zmq_POPULATED)
        FetchContent_Populate(zmq)
        if(POLICY CMP0077)  # Avoid building shared library and tests on CMake 3.13+
            cmake_policy(SET CMP0077 NEW)
            set(BUILD_SHARED OFF CACHE BOOL "")
            set(BUILD_TESTS OFF CACHE BOOL "")
        endif()
        add_subdirectory(${zmq_SOURCE_DIR} ${zmq_BINARY_DIR})
    endif()
    target_link_libraries(ps PRIVATE libzmq-static)
else()
    target_include_directories(ps PRIVATE ${ZMQ_INCLUDE_DIRS})
    target_link_libraries(ps PRIVATE ${ZMQ_LIBRARIES})
endif()

# find and build protobuf
find_package(Protobuf)
if(NOT Protobuf_FOUND)
    message(FATAL_ERROR "Protobuf not found, use `conda install protobuf`")
endif()

set(PROTOBUF_GENERATE_CPP_APPEND_PATH TRUE)
file(GLOB_RECURSE PROTO_FILES "proto/*.proto")
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_FILES})
target_sources(ps PRIVATE ${PROTO_SRCS})
target_include_directories(ps PRIVATE ${CMAKE_CURRENT_BINARY_DIR})  # PROTO_HDRS files are here
target_include_directories(ps PRIVATE ${PROTOBUF_INCLUDE_DIR})
target_include_directories(ps PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(ps PRIVATE ${PROTOBUF_LIBRARY})
