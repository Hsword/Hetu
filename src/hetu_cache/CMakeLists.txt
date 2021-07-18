aux_source_directory(src HETU_SRC_LIST)

find_package(pybind11 2.6.0 CONFIG)

if (NOT pybind11_FOUND)
    message(FATAL_ERROR "pybind11 not found")
else()
    pybind11_add_module(hetu_cache ${HETU_SRC_LIST})
    target_include_directories(hetu_cache PUBLIC include)
    target_link_libraries(hetu_cache PUBLIC ps)
endif()
