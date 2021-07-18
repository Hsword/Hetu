#pragma once

#include "cache.h"

#include <list>
#include <unordered_map>

namespace hetu {

/*
  LRUCache:
    use LRU policy
    Implemented with a double-linked list and a hash map
    O(1) insert, lookup
*/

class LRUCache : public CacheBase {
private:
    std::unordered_map<cache_key_t, std::list<EmbeddingPT>::iterator> hash_;
    std::list<EmbeddingPT> list_;

public:
    using CacheBase::CacheBase;
    size_t size() final {
        return hash_.size();
    }
    int count(cache_key_t k) final;
    void insert(EmbeddingPT e) final;
    EmbeddingPT lookup(cache_key_t k) final;

    // python debug function
    py::array_t<cache_key_t> PyAPI_keys();
}; // class LRUCache

} // namespace hetu
