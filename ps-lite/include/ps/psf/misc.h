#pragma once

#include "PSFunc.h"

namespace ps {

enum InitType {
    Constant,
    Uniform,
    Normal,
    TruncatedNormal,
};

template <>
struct PSFData<ParamInit> {
    static constexpr PsfGroup group = PsfGroup::kParameterServer;
    static constexpr const char* name = "ParamInit";
    using Request = tuple<Key,                // key
                          int,                // param_type
                          size_t,             // len
                          size_t,             // width
                          int,                // init_type
                          double,             // init_a
                          double,             // init_b
                          unsigned long long, // seed
                          int,                // opt_type
                          SArray<float>       // opt arguments
                          >;
    using Response = tuple<>;
    static void _callback(const Response &response) {
    }
};

template <>
struct PSFData<ParamClear> {
    static constexpr PsfGroup group = PsfGroup::kParameterServer;
    static constexpr const char* name = "ParamClear";
    using Request = tuple<Key // key
                          >;
    using Response = tuple<>;
    static void _callback(const Response &response) {
    }
};

template <>
struct PSFData<ParamSave> {
    static constexpr PsfGroup group = PsfGroup::kParameterServer;
    static constexpr const char* name = "ParamSave";
    using Request = tuple<Key,
                          SArray<char>, // address
                          bool          // different from load
                          >;
    using Response = tuple<>;
    static void _callback(const Response &response) {
    }
};

template <>
struct PSFData<ParamLoad> {
    static constexpr PsfGroup group = PsfGroup::kParameterServer;
    static constexpr const char* name = "ParamLoad";
    using Request = tuple<Key,
                          SArray<char> // address
                          >;
    using Response = tuple<>;
    static void _callback(const Response &response) {
    }
};

} // namespace ps
