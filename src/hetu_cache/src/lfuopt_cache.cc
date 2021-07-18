#include "lfuopt_cache.h"

namespace hetu {

int LFUOptCache::count(cache_key_t k) {
    return store_.count(k) + hash_.count(k);
}

void LFUOptCache::insert(EmbeddingPT e) {
    if (store_.count(e->key())) {
        store_[e->key()] = e;
        return;
    }
    auto iter = hash_.find(e->key());
    if (iter != hash_.end()) {
        iter->second->ptr = e;
    } else {
        if (size() == limit_) {
            if (hash_.size() > 0)
                _evict();
            else
                return;
        }
        hash_[e->key()] = _create(e);
    }
}

EmbeddingPT LFUOptCache::lookup(cache_key_t k) {
    auto ptr = store_.find(k);
    if (ptr != store_.end())
        return ptr->second;
    auto iter = hash_.find(k);
    if (iter == hash_.end())
        return nullptr;
    auto result = iter->second->ptr;
    if (iter->second->use + 1 < kUseCntMax)
        hash_[k] = _increase(iter->second);
    else {
        store_[k] = iter->second->ptr;
        clist[kUseCntMax - 1].erase(iter->second);
        hash_.erase(k);
    }
    return result;
}

std::list<LFUOptCache::Block>::iterator
LFUOptCache::_create(EmbeddingPT embed) {
    clist[0].push_front({embed, 0});
    return clist[0].begin();
}

void LFUOptCache::_evict() {
    for (int i = 0; i < kUseCntMax; i++) {
        if (!clist[i].empty()) {
            auto embed = clist[i].back().ptr;
            hash_.erase(embed->key());
            clist[i].pop_back();
            if (embed->getUpdates())
                evict_.push_back(embed);
            break;
        }
    }
}

std::list<LFUOptCache::Block>::iterator
LFUOptCache::_increase(std::list<Block>::iterator iter) {
    size_t use = iter->use;
    clist[use + 1].push_front({iter->ptr, iter->use + 1});
    clist[use].erase(iter);
    return clist[use + 1].begin();
}

py::array_t<cache_key_t> LFUOptCache::PyAPI_keys() {
    std::vector<cache_key_t> keys;
    for (auto &iter : store_)
        keys.push_back(iter.first);
    for (auto &iter : hash_)
        keys.push_back(iter.first);
    std::sort(keys.begin(), keys.end());
    return bind::vec(keys);
}

} // namespace hetu
