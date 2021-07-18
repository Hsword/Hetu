#pragma once

#include "cache.h"

#include <list>
#include <unordered_map>
#include <map>

namespace hetu {

/*
  LFUOptCache:
    use LFU policy, move to permanent store when freq reach n
    Implemented with hashmap and a 2-D list ordered by frequency
    O(1) insert and lookup
*/

class LFUOptCache : public CacheBase {
private:
    struct Block {
        EmbeddingPT ptr;
        int use;
    };
    typedef std::list<Block> CountList;
    const static int kUseCntMax = 10;
    CountList clist[kUseCntMax];
    std::unordered_map<cache_key_t, std::list<Block>::iterator> hash_;

    // helper function
    std::list<Block>::iterator _increase(std::list<Block>::iterator);
    std::list<Block>::iterator _create(EmbeddingPT);
    void _evict();

    std::unordered_map<cache_key_t, EmbeddingPT> store_;

public:
    using CacheBase::CacheBase;
    size_t size() final {
        return store_.size() + hash_.size();
    }
    int count(cache_key_t k) final;
    void insert(EmbeddingPT e) final;
    EmbeddingPT lookup(cache_key_t k) final;

    // python debug function
    py::array_t<cache_key_t> PyAPI_keys();
}; // class LFUCache

} // namespace hetu
