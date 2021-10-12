#include "ps/server/preduce_handler.h"
#include <algorithm>

namespace ps {

void PSHandler<PsfGroup::kPReduceScheduler>::serve(
    const PSFData<kPReduceGetPartner>::Request &request,
    PSFData<kPReduceGetPartner>::Response &response) {
    Key k = get<0>(request);
    int rank = get<1>(request);
    size_t wait_worker_num = get<2>(request);
    float wait_time = get<3>(request);

    // get or initilize the reducestat
    map_mtx_.lock();
    if (!map_.count(k))
        map_.emplace(k, std::unique_ptr<ReduceStat>(new ReduceStat()));
    std::unique_ptr<ReduceStat> &obj = map_[k];
    map_mtx_.unlock();

    std::unique_lock<std::mutex> lock(obj->mtx);
    // must wait until the previous partial reduce decision finish
    while (obj->critical_count) obj->cv.wait(lock);

    if (obj->ready_workers.empty()) {
        // the first worker should set the wait time
        obj->wake_time = std::chrono::system_clock::now() +
            std::chrono::microseconds(int(wait_time * 1000));
    }
    obj->ready_workers.push_back(rank);
    if (obj->ready_workers.size() == wait_worker_num) {
        // if worker number is enough, notify all
        obj->cv.notify_all();
    } else {
        while (obj->ready_workers.size() < wait_worker_num &&
               obj->cv.wait_until(lock, obj->wake_time) == std::cv_status::no_timeout) {}
    }
    // the first worker awake set the critical count
    if (!obj->critical_count) {
        obj->critical_count = obj->ready_workers.size();
        std::sort(obj->ready_workers.begin(), obj->ready_workers.end());
    }

    // write return value
    assert(obj->ready_workers.size() > 0);
    auto &result = get<0>(response);
    result.CopyFrom(obj->ready_workers.data(), obj->ready_workers.size());
    obj->critical_count--;

    // if being the last thread, clear the state
    if (!obj->critical_count) {
        obj->ready_workers.clear();
        obj->cv.notify_all();
    }
    return;
}


} // namespace ps

