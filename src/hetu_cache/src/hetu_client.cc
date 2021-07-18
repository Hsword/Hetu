#include "hetu_client.h"
#include <iostream>

namespace hetu {

size_t syncEmbedding(int node_id, vector<EmbeddingPT> &embed, size_t bound) {
    if (embed.empty())
        return 0;
    size_t n = embed.size();
    size_t width = embed[0]->size();
    SArray<cache_key_t> keys(n);
    SArray<version_t> vers(n);
    for (size_t i = 0; i < n; i++) {
        keys[i] = embed[i]->key();
        vers[i] = embed[i]->getVersion();
    }
    auto pulled = std::make_shared<size_t>();
    auto closure =
        [embed, width,
         pulled](const ps::PSFData<ps::kSyncEmbedding>::Response &response,
                 size_t offset) {
            auto &ret_idx = std::get<0>(response);
            auto &ret_vers = std::get<1>(response);
            auto &value = std::get<2>(response);
            for (size_t i = 0; i < ret_idx.size(); i++) {
                size_t idx = ret_idx[i] + offset;
                embed[idx]->setVersion(ret_vers[i]);
                std::copy(&value[i * width], &value[(i + 1) * width],
                          embed[idx]->data());
                embed[idx]->addup();
            }
            *pulled = ret_idx.size();
        };
    ps::syncEmbedding(node_id, keys, vers, bound, closure);

    return *pulled;
}

void pushEmbedding(int node_id, vector<EmbeddingPT> &embed) {
    if (embed.empty())
        return;
    size_t n = embed.size();
    size_t width = embed[0]->size();
    SArray<cache_key_t> keys(n);
    for (size_t i = 0; i < n; i++)
        keys[i] = embed[i]->key();
    SArray<embed_t> value(n * width);
    SArray<version_t> updates(n);
    for (size_t i = 0; i < n; i++) {
        std::copy(embed[i]->grad(), embed[i]->grad() + width,
                  &value[i * width]);
        updates[i] = embed[i]->getUpdates();
    }
    ps::PushEmbedding(node_id, keys, value, updates);
}

size_t pushSyncEmbedding(int node_id, vector<EmbeddingPT> &embed, size_t bound,
                         vector<EmbeddingPT> &push_embed) {
    if (embed.empty() && push_embed.empty())
        return 0;
    size_t n = embed.size();
    size_t width = embed[0]->size();
    SArray<cache_key_t> keys(n);
    SArray<version_t> vers(n);
    for (size_t i = 0; i < n; i++) {
        keys[i] = embed[i]->key();
        vers[i] = embed[i]->getVersion();
    }
    auto pulled = std::make_shared<size_t>();
    auto closure =
        [embed, width,
         pulled](const ps::PSFData<ps::kSyncEmbedding>::Response &response,
                 size_t offset) {
            auto &ret_idx = std::get<0>(response);
            auto &ret_vers = std::get<1>(response);
            auto &value = std::get<2>(response);
            for (size_t i = 0; i < ret_idx.size(); i++) {
                size_t idx = ret_idx[i] + offset;
                embed[idx]->setVersion(ret_vers[i]);
                std::copy(&value[i * width], &value[(i + 1) * width],
                          embed[idx]->data());
                embed[idx]->addup();
            }
            *pulled = ret_idx.size();
        };

    // Handle Push Embedding
    n = push_embed.size();
    SArray<cache_key_t> push_keys(n);
    for (size_t i = 0; i < n; i++)
        push_keys[i] = push_embed[i]->key();
    SArray<embed_t> value(n * width);
    SArray<version_t> updates(n);
    for (size_t i = 0; i < n; i++) {
        std::copy(push_embed[i]->grad(), push_embed[i]->grad() + width,
                  &value[i * width]);
        updates[i] = push_embed[i]->getUpdates();
    }
    ps::PushSyncEmbedding(node_id, keys, vers, bound, closure, push_keys, value,
                          updates);

    return *pulled;
}

} // namespace hetu
