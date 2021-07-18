#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <vector>

namespace py = pybind11;

namespace hetu {
namespace bind {

// snippet for converting std::vector
template <typename T>
py::array_t<T> vec(std::vector<T> &v) {
    return py::array_t<T>(v.size(), v.data());
}

// snippet for converting std::vector (but without copy)
template <typename T>
py::array_t<T> vec_nocp(std::vector<T> &v) {
    return py::array_t<T>(v.size(), v.data(), py::none());
}

// snippet for converting raw pointer
template <typename T>
py::array_t<T> pt1d(const T *v, size_t cnt) {
    auto result = py::array_t<T>(cnt, v);
    return result;
}

// snippet for converting raw pointer (but without copy)
template <typename T>
py::array_t<T> pt1d_nocp(const T *v, size_t cnt) {
    auto result = py::array_t<T>(cnt, v, py::none());
    return result;
}

// snippet for converting python array
template <typename T>
std::vector<T> a2v(py::array_t<T> &arr) {
    std::vector<T> v(arr.size());
    memcpy(v.data(), arr.data(), v.size() * sizeof(T));
    return v;
}

} // namespace bind

// Check an array is continuous is C
// so that we can confidently use its pointer
#define PYTHON_CHECK_ARRAY(array)                                              \
    {                                                                          \
        if (!array.attr("flags").attr("c_contiguous").cast<bool>()) {          \
            std::string err = "Array not continuous in C: ";                   \
            throw std::runtime_error(err + #array);                            \
        }                                                                      \
    }

} // namespace hetu
