#pragma once

#include "embedding.h"
#include "binding.h"
#include "common/sarray.h"

#include <future>
#include <vector>
using std::vector;

namespace hetu {

typedef std::shared_future<void> wait_t;

/*
  CacheBase:
    CacheBase is the Base class of all cache Policy
    args:
      limit: the number of embedding will not exceed limit
*/
class CacheBase {
protected:
    size_t limit_;
    size_t len_, width_;
    version_t pull_bound_ = 5;
    version_t push_bound_ = 5;
    int node_id_;
    bool bypass_cache_ = false;
    vector<EmbeddingPT> evict_;
    std::mutex mtx;
    bool perf_enabled_ = false;
    py::list perf_;

public:
    /*
      limit: cache size limit
      len: embedding table length
      width: embedding table width
      node_id: the server key
    */
    CacheBase(size_t limit, size_t len, size_t width, int node_id);
    ~CacheBase() {
    }
    size_t getLimit() {
        return limit_;
    }
    //------------------------- cache policy virtual function
    //---------------------
    virtual size_t size() = 0;
    virtual int count(cache_key_t k) = 0;
    virtual void insert(EmbeddingPT e) = 0;
    virtual EmbeddingPT lookup(cache_key_t k) = 0;
    //------------------------- implement tools ---------------------
    // Used to lookup/insert many keys together
    vector<EmbeddingPT> batchedLookup(const cache_key_t *, size_t len);
    void batchedInsert(vector<EmbeddingPT> &ptrs);
    //------------------------- implement main python API ---------------------
    version_t getPullBound() {
        return pull_bound_;
    }
    version_t getPushBound() {
        return push_bound_;
    }
    size_t getWidth() {
        return width_;
    }
    void setPullBound(version_t bound) {
        pull_bound_ = bound;
    }
    void setPushBound(version_t bound) {
        push_bound_ = bound;
    }
    void bypass() {
        bypass_cache_ = true;
    }
    void undoBypass() {
        bypass_cache_ = false;
    }
    bool getPerfEnabled() {
        return perf_enabled_;
    }
    void setPerfEnabled(bool value) {
        perf_enabled_ = value;
    }
    /*
      embeddingLookup is called before each training batch
      * keys may be duplicated, unique operation is required before sending
      server requests
    */
    wait_t embeddingLookup(py::array_t<cache_key_t> keys,
                           py::array_t<embed_t> dest);
    wait_t embeddingLookupRaw(uint64_t _keys, uint64_t dest, size_t num_keys);
    void _embeddingLookup(SArray<cache_key_t> keys, embed_t *dest);
    wait_t embeddingUpdate(py::array_t<cache_key_t> keys,
                           py::array_t<embed_t> grads);
    wait_t embeddingUpdateRaw(uint64_t _keys, uint64_t grads, size_t num_keys);
    void _embeddingUpdate(SArray<cache_key_t> keys, const embed_t *grads);
    wait_t embeddingPushPullRaw(uint64_t _pullkeys, uint64_t _dest,
                                size_t num_pull_keys, uint64_t _pushkeys,
                                uint64_t _grads, size_t num_push_keys);
    void _embeddingPushPull(SArray<cache_key_t> keys, embed_t *dest,
                            SArray<cache_key_t> push_keys,
                            const embed_t *grads);
    std::string __repr__();
    py::list getPerf() {
        return perf_;
    };
}; // class CacheBase

} // namespace hetu
