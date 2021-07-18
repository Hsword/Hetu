#include <chrono>

#include "cache.h"
#include "hetu_client.h"
#include "unqiue_tools.h"
#include "common/thread_pool.h"

namespace hetu {

CacheBase::CacheBase(size_t limit, size_t len, size_t width, int node_id) :
    limit_(limit), width_(width), node_id_(node_id) {
}

vector<EmbeddingPT> CacheBase::batchedLookup(const cache_key_t *keys,
                                             size_t len) {
    std::lock_guard<std::mutex> lock(mtx);
    if (bypass_cache_)
        return vector<EmbeddingPT>(len, nullptr);
    vector<EmbeddingPT> result(len);
    for (size_t i = 0; i < len; i++) {
        EmbeddingPT ptr = lookup(keys[i]);
        result[i] = ptr;
    }
    return result;
}

void CacheBase::batchedInsert(vector<EmbeddingPT> &ptrs) {
    std::lock_guard<std::mutex> lock(mtx);
    if (bypass_cache_)
        return;
    for (auto &ptr : ptrs) {
        insert(ptr);
    }
}

wait_t CacheBase::embeddingLookup(py::array_t<cache_key_t> _keys,
                                  py::array_t<embed_t> _dest) {
    PYTHON_CHECK_ARRAY(_keys);
    PYTHON_CHECK_ARRAY(_dest);
    auto dest = _dest.mutable_data();
    size_t num_keys = _keys.size();
    assert(size_t(_dest.size()) == num_keys * width_);
    SArray<cache_key_t> keys(_keys.mutable_data(), num_keys);
    return ThreadPool::Get()->Enqueue(&CacheBase::_embeddingLookup, this, keys,
                                      dest);
}

wait_t CacheBase::embeddingLookupRaw(uint64_t _keys, uint64_t dest,
                                     size_t num_keys) {
    float *keys = reinterpret_cast<float *>(_keys);
    SArray<cache_key_t> intkeys(num_keys);
    for (size_t i = 0; i < num_keys; i++)
        intkeys[i] = (cache_key_t)keys[i];
    return ThreadPool::Get()->Enqueue(&CacheBase::_embeddingLookup, this,
                                      intkeys,
                                      reinterpret_cast<embed_t *>(dest));
}

void CacheBase::_embeddingLookup(SArray<cache_key_t> keys, embed_t *dest) {
    auto start_time = std::chrono::system_clock::now();
    // Unique operation
    auto unique_keys = Unique<cache_key_t>(keys.data(), keys.size());
    auto unique_time = std::chrono::system_clock::now();
    // Lookup all the keys together
    auto embeds = batchedLookup(unique_keys.data(), unique_keys.size());
    auto lookup_time = std::chrono::system_clock::now();

    // Scan out missed keys and pull from server
    vector<EmbeddingPT> should_insert;
    for (size_t i = 0; i < unique_keys.size(); i++) {
        if (!embeds[i]) {
            embeds[i].reset(new Embedding(unique_keys[i], width_));
            should_insert.push_back(embeds[i]);
        }
    }
    auto mem_time = std::chrono::system_clock::now();
    size_t pulled = syncEmbedding(node_id_, embeds, pull_bound_);
    auto trans_time = std::chrono::system_clock::now();
    // Copy embedding to destination
    for (size_t _i = 0; _i < keys.size(); _i++) {
        auto i = unique_keys.map(_i);
        std::copy(embeds[i]->data(), embeds[i]->data() + embeds[i]->size(),
                  dest + _i * embeds[i]->size());
    }
    auto copy_time = std::chrono::system_clock::now();
    batchedInsert(should_insert);
    auto end_time = std::chrono::system_clock::now();
    if (perf_enabled_) {
        py::gil_scoped_acquire acquire;
        py::dict performance;
        performance["type"] = "Pull";
        performance["is_full"] = size() == limit_;
        performance["num_all"] = keys.size();
        performance["num_unique"] = unique_keys.size();
        performance["num_miss"] = should_insert.size();
        performance["num_transfered"] = pulled;
        performance["time"] = (end_time - start_time).count() / 1e6;
        performance["sort_time"] = (unique_time - start_time).count() / 1e6;
        performance["lookup_time"] = (lookup_time - unique_time).count() / 1e6;
        performance["prepare_time"] = (mem_time - lookup_time).count() / 1e6;
        performance["transfer_time"] = (trans_time - mem_time).count() / 1e6;
        performance["copy_time"] = (copy_time - trans_time).count() / 1e6;
        performance["insert_time"] = (end_time - copy_time).count() / 1e6;
        perf_.append(performance);
    }
}

wait_t CacheBase::embeddingUpdate(py::array_t<cache_key_t> _keys,
                                  py::array_t<embed_t> _grads) {
    PYTHON_CHECK_ARRAY(_keys);
    PYTHON_CHECK_ARRAY(_grads);
    auto grads = _grads.data();
    size_t num_keys = _keys.size();
    assert(size_t(_grads.size()) == num_keys * width_);
    SArray<cache_key_t> keys(_keys.mutable_data(), num_keys);
    return ThreadPool::Get()->Enqueue(&CacheBase::_embeddingUpdate, this, keys,
                                      grads);
}

wait_t CacheBase::embeddingUpdateRaw(uint64_t _keys, uint64_t grads,
                                     size_t num_keys) {
    float *keys = reinterpret_cast<float *>(_keys);
    SArray<cache_key_t> intkeys(num_keys);
    for (size_t i = 0; i < num_keys; i++)
        intkeys[i] = (cache_key_t)keys[i];
    return ThreadPool::Get()->Enqueue(&CacheBase::_embeddingUpdate, this,
                                      intkeys,
                                      reinterpret_cast<embed_t *>(grads));
}

void CacheBase::_embeddingUpdate(SArray<cache_key_t> keys,
                                 const embed_t *grads) {
    auto start_time = std::chrono::system_clock::now();
    // Unique operation
    auto unique_keys = Unique<cache_key_t>(keys.data(), keys.size());
    auto unique_time = std::chrono::system_clock::now();
    // Lookup all the keys together
    auto embeds = batchedLookup(unique_keys.data(), unique_keys.size());
    auto lookup_time = std::chrono::system_clock::now();
    // Do local updates
    size_t miss_cnt = 0, evict_cnt = evict_.size();
    vector<EmbeddingPT> should_push;
    vector<EmbeddingPT> evict = std::move(evict_);
    for (size_t _i = 0; _i < keys.size(); _i++) {
        auto i = unique_keys.map(_i);
        if (!embeds[i]) {
            // !! This is not likely to happen, newly pulled embedding should be
            // in cache
            embeds[i].reset(new Embedding(unique_keys[i], width_, false));
            miss_cnt++;
        }
        embeds[i]->accumulate(grads + _i * width_);
    }
    auto evict_iter = evict.begin();
    for (size_t i = 0; i < unique_keys.size(); i++) {
        if (embeds[i]->getUpdates() > push_bound_ || !embeds[i]->data()) {
            // merge sort
            while (evict_iter != evict.end()
                   && (*evict_iter)->key() < embeds[i]->key())
                should_push.push_back(*evict_iter++);
            should_push.push_back(embeds[i]);
        }
    }
    while (evict_iter != evict.end())
        should_push.push_back(*evict_iter++);
    auto accum_time = std::chrono::system_clock::now();
    pushEmbedding(node_id_, should_push);
    // After push do some clean up
    auto trans_time = std::chrono::system_clock::now();
    for (size_t i = 0; i < unique_keys.size(); i++) {
        if (embeds[i]->getUpdates() > push_bound_ && embeds[i]->data()) {
            embeds[i]->setVersion(embeds[i]->getVersion()
                                  + embeds[i]->getUpdates());
            embeds[i]->zeroGrad();
        }
    }
    auto end_time = std::chrono::system_clock::now();
    if (perf_enabled_) {
        py::gil_scoped_acquire acquire;
        py::dict performance;
        performance["type"] = "Push";
        performance["is_full"] = size() == limit_;
        performance["num_all"] = keys.size();
        performance["num_unique"] = unique_keys.size();
        performance["num_evict"] = evict_cnt;
        performance["num_miss"] = miss_cnt;
        performance["num_transfered"] = should_push.size();
        performance["time"] = (end_time - start_time).count() / 1e6;
        performance["sort_time"] = (unique_time - start_time).count() / 1e6;
        performance["lookup_time"] = (lookup_time - unique_time).count() / 1e6;
        performance["copy_time"] = (accum_time - lookup_time).count() / 1e6;
        performance["transfer_time"] = (trans_time - accum_time).count() / 1e6;
        performance["cleanup_time"] = (end_time - trans_time).count() / 1e6;
        perf_.append(performance);
    }
}

wait_t CacheBase::embeddingPushPullRaw(uint64_t _pullkeys, uint64_t _dest,
                                       size_t num_pull_keys, uint64_t _pushkeys,
                                       uint64_t _grads, size_t num_push_keys) {
    float *pullkeys = reinterpret_cast<float *>(_pullkeys);
    float *pushkeys = reinterpret_cast<float *>(_pushkeys);
    SArray<cache_key_t> intpullkeys(num_pull_keys), intpushkeys(num_push_keys);
    for (size_t i = 0; i < num_pull_keys; i++)
        intpullkeys[i] = (cache_key_t)pullkeys[i];
    for (size_t i = 0; i < num_push_keys; i++)
        intpushkeys[i] = (cache_key_t)pushkeys[i];
    return ThreadPool::Get()->Enqueue([this, intpullkeys, intpushkeys, _grads,
                                       _dest]() {
        // _embeddingUpdate(intpushkeys, reinterpret_cast<embed_t*>(_grads));
        // _embeddingLookup(intpullkeys, reinterpret_cast<embed_t*>(_dest));
        _embeddingPushPull(intpullkeys, reinterpret_cast<embed_t *>(_dest),
                           intpushkeys, reinterpret_cast<embed_t *>(_grads));
    });
}

void CacheBase::_embeddingPushPull(SArray<cache_key_t> keys, embed_t *dest,
                                   SArray<cache_key_t> push_keys,
                                   const embed_t *grads) {
    auto unique_keys = Unique<cache_key_t>(keys.data(), keys.size());
    // Lookup all the keys together
    auto embeds = batchedLookup(unique_keys.data(), unique_keys.size());
    // Scan out missed keys and pull from server
    vector<EmbeddingPT> should_insert;
    for (size_t i = 0; i < unique_keys.size(); i++) {
        if (!embeds[i]) {
            embeds[i].reset(new Embedding(unique_keys[i], width_));
            should_insert.push_back(embeds[i]);
        }
    }

    auto push_unique_keys =
        Unique<cache_key_t>(push_keys.data(), push_keys.size());
    auto push_embeds =
        batchedLookup(push_unique_keys.data(), push_unique_keys.size());
    size_t miss_cnt = 0;
    // size_t evict_cnt = evict_.size();
    vector<EmbeddingPT> should_push;
    vector<EmbeddingPT> evict = std::move(evict_);
    for (size_t _i = 0; _i < push_keys.size(); _i++) {
        auto i = push_unique_keys.map(_i);
        if (!push_embeds[i]) {
            // !! This is not likely to happen, newly pulled embedding should be
            // in cache
            push_embeds[i].reset(
                new Embedding(push_unique_keys[i], width_, false));
            miss_cnt++;
        }
        push_embeds[i]->accumulate(grads + _i * width_);
    }
    auto evict_iter = evict.begin();
    for (size_t i = 0; i < push_unique_keys.size(); i++) {
        if (push_embeds[i]->getUpdates() > push_bound_
            || !push_embeds[i]->data()) {
            // merge sort
            while (evict_iter != evict.end()
                   && (*evict_iter)->key() < push_embeds[i]->key())
                should_push.push_back(*evict_iter++);
            should_push.push_back(push_embeds[i]);
        }
    }
    while (evict_iter != evict.end())
        should_push.push_back(*evict_iter++);
    // size_t pulled =
    pushSyncEmbedding(node_id_, embeds, pull_bound_, should_push);

    // Copy embedding to destination
    for (size_t _i = 0; _i < keys.size(); _i++) {
        auto i = unique_keys.map(_i);
        std::copy(embeds[i]->data(), embeds[i]->data() + embeds[i]->size(),
                  dest + _i * embeds[i]->size());
    }
    batchedInsert(should_insert);

    for (size_t i = 0; i < push_unique_keys.size(); i++) {
        if (push_embeds[i]->getUpdates() > push_bound_
            && push_embeds[i]->data()) {
            push_embeds[i]->setVersion(push_embeds[i]->getVersion()
                                       + push_embeds[i]->getUpdates());
            push_embeds[i]->zeroGrad();
        }
    }
}

std::string CacheBase::__repr__() {
    std::stringstream ss;
    ss << "<Cache : ";
    ss << size() << "/" << limit_;
    ss << " , id:" << node_id_;
    ss << " , width:" << width_;
    ss << " , bound:" << pull_bound_ << " " << push_bound_;
    ss << ">";
    return ss.str();
}

} // namespace hetu
