#pragma once

#include "ps/psf/PSFunc.h"

#include <unordered_map>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <chrono>

namespace ps {
template<>
class PSHandler<PsfGroup::kPReduceScheduler> : public PSHandler<PsfGroup::kBaseGroup> {
public:
    PSHandler<PsfGroup::kPReduceScheduler>() {
    }
    PSHandler<PsfGroup::kPReduceScheduler>(const PSHandler<PsfGroup::kPReduceScheduler> &handle) {
    }

    void serve(const PSFData<kPReduceGetPartner>::Request &request,
               PSFData<kPReduceGetPartner>::Response &response);

private:
    struct ReduceStat {
        std::mutex mtx;
        std::vector<int> ready_workers;
        std::condition_variable cv;
        decltype(std::chrono::system_clock::now()) wake_time;
        int critical_count = 0; // stop new worker from coming in, when the previous schedule is finishing
    }; // store the state for every reduce key

    std::unordered_map<Key, std::unique_ptr<ReduceStat>> map_;
    std::mutex map_mtx_; // lock for the map
};

} // namespace ps
