# - Try to find CUB
# Once done this will define
# CUB_FOUND - System has CUB
# CUB_INCLUDE_DIR - The CUB include directories

find_path ( CUB_INCLUDE_DIR cub HINTS ${CUB_ROOT}/include )

find_package_handle_standard_args (
    CUB
    REQUIRED_VARS CUB_INCLUDE_DIR)
