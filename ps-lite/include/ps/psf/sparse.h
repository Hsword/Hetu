#pragma once

#include "PSFunc.h"
#include "dense.h"

namespace ps {

template <>
struct PSFData<SparsePull> {
    static constexpr PsfGroup group = PsfGroup::kParameterServer;
    static constexpr const char* name = "SparsePull";
    using Request = tuple<Key,           // key
                          SArray<size_t> // offset
                          >;
    using Response = tuple<SArray<float> // data
                           >;
    static void
    _callback(const Response &response, SArray<float> tgt,
              std::vector<std::pair<size_t, std::vector<size_t>>> mapping,
              size_t offset, size_t width) {
        auto val = get<0>(response);
        CHECK_EQ(val.size(), mapping.size() * width)
            << val.size() << " " << mapping.size() << " " << width;
        for (size_t i = 0; i < mapping.size(); ++i) {
            auto begin_iter = val.begin() + i * width;
            auto end_iter = begin_iter + width;
            for (auto idx : mapping[i].second) {
                std::copy(begin_iter, end_iter, tgt.begin() + idx * width);
            }
        }
    }
};

template <>
struct PSFData<SparsePush> {
    static constexpr PsfGroup group = PsfGroup::kParameterServer;
    static constexpr const char* name = "SparsePush";
    using Request = tuple<Key,            // key
                          SArray<size_t>, // offset
                          SArray<float>   // data
                          >;
    using Response = tuple<>;
    static void _callback(const Response &response) {
    }
};

template <>
struct PSFData<SDPushPull> {
    static constexpr PsfGroup group = PsfGroup::kParameterServer;
    static constexpr const char* name = "SDPushPull";
    using Request = tuple<Key,            // key
                          SArray<size_t>, // offset
                          SArray<float>,  // data
                          size_t          // len for densepull
                          >;
    using Response = PSFData<DensePull>::Response;

    static void _callback(const Response &response, SArray<float> tgt) {
        auto val = get<0>(response);
        CHECK_EQ(val.size(), tgt.size()) << val.size() << " " << tgt.size();
        std::copy(val.begin(), val.end(), tgt.begin());
    }
};

template <>
struct PSFData<SSPushPull> {
    static constexpr PsfGroup group = PsfGroup::kParameterServer;
    static constexpr const char* name = "SSPushPull";
    using Request = tuple<Key,            // key
                          SArray<size_t>, // push offset
                          SArray<float>,  // data
                          SArray<size_t>  // pull offset
                          >;
    using Response = PSFData<SparsePull>::Response;

    static void
    _callback(const Response &response, SArray<float> tgt,
              std::vector<std::pair<size_t, std::vector<size_t>>> mapping,
              size_t offset, size_t width) {
        auto val = get<0>(response);
        if (val.size() > 0) {
            CHECK_EQ(val.size(), mapping.size() * width)
                << val.size() << " " << mapping.size() << " " << width;
            for (size_t i = 0; i < mapping.size(); ++i) {
                auto begin_iter = val.begin() + i * width;
                auto end_iter = begin_iter + width;
                for (auto idx : mapping[i].second) {
                    std::copy(begin_iter, end_iter, tgt.begin() + idx * width);
                }
            }
        }
    }
};

} // namespace ps
