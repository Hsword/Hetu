#pragma once

#include "PSFunc.h"

namespace ps {

template <>
struct PSFData<DensePull> {
    static constexpr PsfGroup group = PsfGroup::kParameterServer;
    static constexpr const char* name = "DensePull";
    using Request = tuple<Key,   // key
                          size_t // len
                          >;
    using Response = tuple<SArray<float> // data
                           >;
    static void _callback(const Response &response, SArray<float> tgt) {
        auto val = get<0>(response);
        CHECK_EQ(val.size(), tgt.size()) << val.size() << " " << tgt.size();
        std::copy(val.begin(), val.end(), tgt.begin());
    }
};

template <>
struct PSFData<DensePush> {
    static constexpr PsfGroup group = PsfGroup::kParameterServer;
    static constexpr const char* name = "DensePush";
    using Request = tuple<Key,          // key
                          size_t,       // len
                          SArray<float> // data
                          >;
    using Response = tuple<>;
    static void _callback(const Response &response) {
    }
};

template <>
struct PSFData<DDPushPull> {
    static constexpr PsfGroup group = PsfGroup::kParameterServer;
    static constexpr const char* name = "DDPushPull";
    using Request = PSFData<DensePush>::Request;
    using Response = PSFData<DensePull>::Response;

    static void _callback(const Response &response, SArray<float> tgt) {
        auto val = get<0>(response);
        CHECK_EQ(val.size(), tgt.size()) << val.size() << " " << tgt.size();
        std::copy(val.begin(), val.end(), tgt.begin());
    }
};

} // namespace ps
