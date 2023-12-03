# - Try to find THRUST
# Once done this will define
# THRUST_FOUND - System has THRUST
# THRUST_INCLUDE_DIR - The THRUST include directories

find_path ( THRUST_INCLUDE_DIR thrust HINTS ${THRUST_ROOT}/include )

find_package_handle_standard_args (
    THRUST
    REQUIRED_VARS THRUST_INCLUDE_DIR)
