#include "ps/worker/worker.h"
#include "ps/worker/hetu_binding.h"

namespace ps {

void syncEmbedding(int node_id, const SArray<uint64_t> &keys,
                   const SArray<version_t> &ver, version_t bound,
                   PSFData<kSyncEmbedding>::Closure closure) {
    PSAgent::Get()->syncEmbedding(node_id, keys, ver, bound, closure);
    PSAgent::Get()->wait(node_id);
}

void PushEmbedding(int node_id, const SArray<uint64_t> &keys,
                   const SArray<float> &data,
                   const SArray<version_t> &updates) {
    PSAgent::Get()->pushEmbedding(node_id, keys, data, updates);
    PSAgent::Get()->wait(node_id);
}

void PushSyncEmbedding(int node_id, const SArray<uint64_t> &keys,
                       const SArray<version_t> &ver, version_t bound,
                       PSFData<kSyncEmbedding>::Closure closure,
                       const SArray<uint64_t> &push_keys,
                       const SArray<float> &data,
                       const SArray<version_t> &updates) {
    PSAgent::Get()->pushSyncEmbedding(node_id, keys, ver, bound, closure,
                                      push_keys, data, updates);
    PSAgent::Get()->wait(node_id);
}

void debug() {
    printf("hetu at %p\n", Postoffice::Get());
}

} // namespace ps
