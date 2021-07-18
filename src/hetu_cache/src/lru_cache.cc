#include "lru_cache.h"

namespace hetu {

int LRUCache::count(cache_key_t k) {
    return hash_.count(k);
}

void LRUCache::insert(EmbeddingPT e) {
    assert(e->size() == width_);
    if (hash_.count(e->key())) {
        auto embed = hash_[e->key()];
        list_.erase(embed);
    }
    list_.push_front(e);
    hash_[e->key()] = list_.begin();
    // Evict the least resently used if exceeds
    if (hash_.size() > limit_) {
        auto embed = list_.back();
        hash_.erase(embed->key());
        if (embed->getUpdates() != 0)
            evict_.push_back(embed);
        list_.pop_back();
    }
}

EmbeddingPT LRUCache::lookup(cache_key_t k) {
    auto iter = hash_.find(k);
    if (iter == hash_.end()) {
        return nullptr;
    }
    auto list_iterator = iter->second;
    auto result = *list_iterator;
    // Move the recently used cache line to the front of the list
    list_.erase(list_iterator);
    list_.push_front(result);
    hash_[k] = list_.begin();
    return result;
}

py::array_t<cache_key_t> LRUCache::PyAPI_keys() {
    std::vector<cache_key_t> keys;
    for (auto &iter : hash_) {
        keys.push_back(iter.first);
    }
    std::sort(keys.begin(), keys.end());
    return bind::vec(keys);
}

} // namespace hetu
