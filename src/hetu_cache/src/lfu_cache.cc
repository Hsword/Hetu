#include "lfu_cache.h"

namespace hetu {

int LFUCache::count(cache_key_t k) {
    return hash_.count(k);
}

void LFUCache::insert(EmbeddingPT e) {
    assert(e->size() == width_);
    auto iter = hash_.find(e->key());
    if (iter == hash_.end()) {
        if (hash_.size() == limit_)
            _evict();
        hash_[e->key()] = _create(e);
    } else {
        iter->second->ptr = e;
        hash_[e->key()] = _increase(iter->second);
    }
}

EmbeddingPT LFUCache::lookup(cache_key_t k) {
    auto iter = hash_.find(k);
    if (iter == hash_.end())
        return nullptr;
    hash_[k] = _increase(iter->second);
    auto result = iter->second->ptr;
    return result;
}

void LFUCache::_evict() {
    auto clist = list_.begin();
    auto embed = clist->list.back().ptr;
    auto key = embed->key();
    if (embed->getUpdates() != 0)
        evict_.push_back(embed);
    hash_.erase(key);
    clist->list.pop_back();
    if (clist->list.empty())
        list_.erase(clist);
}

std::list<LFUCache::Block>::iterator LFUCache::_create(EmbeddingPT embed) {
    if (list_.empty() || list_.begin()->use > 1) {
        list_.push_front({std::list<Block>(), 1});
    }
    list_.begin()->list.push_front({embed, list_.begin()});
    return list_.begin()->list.begin();
}

std::list<LFUCache::Block>::iterator
LFUCache::_increase(std::list<Block>::iterator iter) {
    std::list<Block>::iterator result;
    auto clist = iter->head;
    auto clist_nxt = ++iter->head;
    size_t use = clist->use + 1;
    if (clist_nxt != list_.end() && clist_nxt->use == use) {
        clist_nxt->list.push_front({iter->ptr, clist_nxt});
        result = clist_nxt->list.begin();
    } else {
        CountList temp = {{}, use};
        auto clist_new = list_.emplace(clist_nxt, temp);
        clist_new->list.push_front({iter->ptr, clist_new});
        result = clist_new->list.begin();
    }
    clist->list.erase(iter);
    if (clist->list.empty())
        list_.erase(clist);
    return result;
}

py::array_t<cache_key_t> LFUCache::PyAPI_keys() {
    std::vector<cache_key_t> keys;
    for (auto &iter : hash_) {
        keys.push_back(iter.first);
    }
    std::sort(keys.begin(), keys.end());
    return bind::vec(keys);
}

} // namespace hetu
