#pragma once

#include "PSFunc.h"

namespace ps {

template <>
struct PSFData<kPReduceGetPartner> {
    static constexpr PsfGroup group = PsfGroup::kPReduceScheduler;
    static constexpr const char* name = "PReduceGetPartner";
    using Request = tuple<
        Key, // reduce group key, each pipeline stage has a unique key
        int, // worker rank
        size_t, // desired worker num
        float // max wait time (ms)
    >;
    using Response = tuple<
        SArray<int> // all the partners worker id to do reduce with
    >;
    static void _callback(const Response &response, int* tgt) {
        auto &val = get<0>(response);
        std::copy(val.begin(), val.end(), tgt);
        tgt[val.size()] = -1;
    }
};

} // namespace ps
