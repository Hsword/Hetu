#pragma once

#include "ps/psf/PSFunc.h"

#include <unordered_map>
#include <algorithm>
#include <mutex>

namespace ps {

template<>
class PSHandler<PsfGroup::kSSPControl> : public PSHandler<PsfGroup::kBaseGroup> {
public:
    PSHandler<PsfGroup::kSSPControl>() {
    }
    PSHandler<PsfGroup::kSSPControl>(const PSHandler<PsfGroup::kSSPControl> &handle) {
    }

    void serve(const PSFData<kSSPInit>::Request &request,
               PSFData<kSSPInit>::Response &response) {
        Key k = get<0>(request);
        int rank = get<1>(request);
        size_t group_size = get<2>(request);
        ssp_version_t tol = get<3>(request);

        std::lock_guard<std::mutex> lock(mtx_);
        if (map_.find(k) == map_.end()) {
            map_[k] = SSPInternalState();
            map_[k].tolerance = tol;
            map_[k].group_size = group_size;
        }
        SSPInternalState &state = map_[k];
        CHECK_EQ(state.tolerance, tol) << "kSSPInit : tolerance mismatch";
        CHECK_EQ(state.group_size, group_size) << "kSSPInit : group size mismatch";
        CHECK(!state.rank2version.count(rank)) << "kSSPInit : duplicated init";
        state.rank2version[rank] = 0;
        CHECK_LE(state.rank2version.size(), group_size) << "kSSPInit : group size larger than desired";
        return;
    }

    void serve(const PSFData<kSSPSync>::Request &request,
               PSFData<kSSPSync>::Response &response) {
        Key k = get<0>(request);
        int rank = get<1>(request);
        ssp_version_t version = get<2>(request);
        bool &reply = get<0>(response);

        std::lock_guard<std::mutex> lock(mtx_);

        CHECK(map_.count(k)) << "kSSPSync : cannot find key " << k;
        SSPInternalState &state = map_[k];
        CHECK(state.rank2version.count(rank));
        if (state.rank2version.size() < state.group_size) {
            // not fully initialized
            reply = false;
            return;
        }
        if (state.rank2version[rank] < version) {
            state.rank2version[rank] = version;
        }
        reply = true;
        for (const auto &p : state.rank2version) {
            if (p.second + state.tolerance < version) {
                reply = false;
                break;
            }
        }
        return;
    }

private:
    struct SSPInternalState {
        std::unordered_map<int, ssp_version_t> rank2version;
        ssp_version_t tolerance;
        size_t group_size;
    };
    std::unordered_map<Key, SSPInternalState> map_;
    std::mutex mtx_;

};

} // namespace ps
