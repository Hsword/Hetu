#pragma once

#include "embedding.h"
#include "ps/worker/hetu_binding.h"
#include <vector>

using std::vector;

namespace hetu {

/*
    This function should pull versions and embedding with the keys.
    * sync function call
    * send all the version in embed (some newly created should be -1)
    * server will decide which embedding is outdated and send the embedding back
    * put them back into embed
    return how many embeddings are really pulled
*/
size_t syncEmbedding(int node_id, vector<EmbeddingPT> &embed, size_t bound);

/*
    This function push the gradients for given embedding
    * push gradients in embed[i]->grads
    * should also push updates number
    * not responsible to call embed[i]->zeroGrad()
*/
void pushEmbedding(int node_id, vector<EmbeddingPT> &embed);

size_t pushSyncEmbedding(int node_id, vector<EmbeddingPT> &embed, size_t bound,
                         vector<EmbeddingPT> &push_embed);

} // namespace hetu
