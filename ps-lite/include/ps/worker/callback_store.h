#pragma once

#include "ps/psf/PSFunc.h"

#include "unordered_map"
#include "mutex"

namespace ps {

// Used to lookup the callback for different ps functions
// Store a callback use store(timestamp, cb)
// Run a callback use run(timestamp, response)
template <PsfType ftype>
class CallbackStore {
public:
    using CallBack = function<void(const typename PSFData<ftype>::Response &)>;
    static CallbackStore *Get() {
        static CallbackStore a;
        return &a;
    }
    void run(int timestamp, const typename PSFData<ftype>::Response &response) {
        mu_.lock();
        auto it = store_.find(timestamp);
        if (it != store_.end()) {
            mu_.unlock();
            CHECK(it->second);
            it->second(response);
            mu_.lock();
            store_.erase(it);
        }
        mu_.unlock();
    }
    void store(int ts, const CallBack &cb) {
        mu_.lock();
        store_[ts] = cb;
        mu_.unlock();
    }

private:
    std::unordered_map<int, CallBack> store_;
    std::mutex mu_;
};

} // namespace ps
