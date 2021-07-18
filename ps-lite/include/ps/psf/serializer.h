#pragma once

#include "common/sarray.h"

#include <tuple>
#include <vector>
using std::tuple;
using std::vector;

namespace ps {

// we don't have if-constexpr in c++11, so we use this
template <bool T>
class ScalarTag {};

// decide whether a data is scalar type or SArray
// isScalar<int>::value -> true
template <typename T>
class isScalar {
public:
    constexpr static bool value =
        std::is_integral<T>::value || std::is_floating_point<T>::value;
    using Tag = ScalarTag<value>;
};

// Helper class to serialize Tuples recursively
template <typename Tuple, int N>
class tupleSerializer {
public:
    // encode scalar type, put it in target[0]
    template <typename dtype>
    static void _encode(const dtype &t, vector<SArray<char>> &target,
                        ScalarTag<true>) {
        size_t cur_size = target[0].size();
        target[0].resize(cur_size + sizeof(dtype));
        dtype *ptr = reinterpret_cast<dtype *>(target[0].data() + cur_size);
        *ptr = t;
    }
    // encode sarray type, append it to target(no copy)
    template <typename dtype>
    static void _encode(const dtype &t, vector<SArray<char>> &target,
                        ScalarTag<false>) {
        SArray<char> bytes(t);
        target.push_back(bytes);
    }
    // encode a tuple from back to front
    static void encode(const Tuple &tup, vector<SArray<char>> &target) {
        auto &t = std::get<N - 1>(tup);
        typedef typename std::remove_reference<decltype(t)>::type dtype;
        _encode(t, target, typename isScalar<dtype>::Tag());
        tupleSerializer<Tuple, N - 1>::encode(tup, target);
    }
    //---------------------------------Decode---------------------------------------
    template <typename dtype>
    static void _decode(dtype &t, const vector<SArray<char>> &target,
                        ScalarTag<true>, size_t &scalar_hint,
                        size_t &array_hint) {
        dtype *ptr = reinterpret_cast<dtype *>(target[0].data() + scalar_hint
                                               - sizeof(dtype));
        t = *ptr;
        scalar_hint -= sizeof(dtype);
    }
    template <typename dtype>
    static void _decode(dtype &t, const vector<SArray<char>> &target,
                        ScalarTag<false>, size_t &scalar_hint,
                        size_t &array_hint) {
        t = target[array_hint - 1];
        array_hint--;
    }
    // scalar_hint, array_hint, tell where to take the data from target
    static void decode(Tuple &tup, const vector<SArray<char>> &target,
                       size_t scalar_hint, size_t array_hint) {
        // When decode, from front to back
        auto &t = std::get<std::tuple_size<Tuple>::value - N>(tup);
        typedef typename std::remove_reference<decltype(t)>::type dtype;
        _decode(t, target, typename isScalar<dtype>::Tag(), scalar_hint,
                array_hint);
        tupleSerializer<Tuple, N - 1>::decode(tup, target, scalar_hint,
                                              array_hint);
    }
};

// Handle template specialization
template <typename Tuple>
class tupleSerializer<Tuple, 0> {
public:
    static void encode(const Tuple &tup, vector<SArray<char>> &target) {
    }
    static void decode(Tuple &tup, const vector<SArray<char>> &target,
                       size_t scalar_hint, size_t array_hint) {
    }
};

// ------------------------------ Exported APIs
// ------------------------------------------------
template <typename Tuple>
void tupleEncode(const Tuple &tup, vector<SArray<char>> &dest) {
    dest.clear();
    dest.push_back(SArray<char>()); // Reserve for scalar types
    dest[0].reserve(sizeof(Tuple));
    tupleSerializer<Tuple, std::tuple_size<Tuple>::value>::encode(tup, dest);
}

template <typename Tuple>
void tupleDecode(Tuple &tup, const vector<SArray<char>> &dest) {
    tupleSerializer<Tuple, std::tuple_size<Tuple>::value>::decode(
        tup, dest, dest[0].size(), dest.size());
}

} // namespace ps
