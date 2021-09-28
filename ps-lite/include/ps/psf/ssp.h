#pragma once

#include "PSFunc.h"

namespace ps {

typedef unsigned long long ssp_version_t;

template <>
struct PSFData<kSSPInit> {
    static constexpr PsfGroup group = PsfGroup::kSSPControl;
    static constexpr const char* name = "SSPInit";
    using Request = tuple<
        Key, // SSP group key
        int, // worker rank
        size_t, // SSP group size
        ssp_version_t // SSP staleness tolerance
    >;
    using Response = tuple<>;
    static void _callback(const Response &response) {
    }
};

template <>
struct PSFData<kSSPSync> {
    static constexpr PsfGroup group = PsfGroup::kSSPControl;
    static constexpr const char* name = "SSPSync";
    using Request = tuple<
        Key, // SSP group key
        int, // worker rank
        ssp_version_t // worker's current stage
    >;
    using Response = tuple<bool>;
    static void _callback(const Response &response, bool &ret_val) {
        ret_val = get<0>(response);
    }
};

} // namespace ps
