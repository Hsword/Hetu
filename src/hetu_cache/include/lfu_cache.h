#pragma once

#include "cache.h"

#include <list>
#include <unordered_map>

namespace hetu {

/*
  LFUCache:
    use LFU policy
    Implemented with hashmap and a 2-D list ordered by frequency
    O(1) insert and lookup
*/

class LFUCache : public CacheBase {
private:
    struct CountList;
    struct Block {
        EmbeddingPT ptr;
        std::list<CountList>::iterator head;
    };
    struct CountList {
        std::list<Block> list;
        size_t use;
    };
    std::list<CountList> list_;
    std::unordered_map<cache_key_t, std::list<Block>::iterator> hash_;

    // helper function
    std::list<Block>::iterator _increase(std::list<Block>::iterator);
    std::list<Block>::iterator _create(EmbeddingPT);
    void _evict();

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
}; // class LFUCache

} // namespace hetu
