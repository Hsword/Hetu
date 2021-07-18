#pragma once

// Do not include worker.h or any ps-lite header here
// or we will have multiple PSAgent, PostOffice instance

#include <common/sarray.h>
#include "ps/psf/PSFunc.h"
using std::vector;

namespace ps {

void debug();

void syncEmbedding(int node_id, const SArray<uint64_t> &keys,
                   const SArray<version_t> &ver, version_t bound,
                   PSFData<kSyncEmbedding>::Closure closure);

// Push Grads and Updates
// keys are unique
void PushEmbedding(int node_id, const SArray<uint64_t> &keys,
                   const SArray<float> &data, const SArray<version_t> &updates);

void PushSyncEmbedding(int node_id, const SArray<uint64_t> &keys,
                       const SArray<version_t> &ver, version_t bound,
                       PSFData<kSyncEmbedding>::Closure closure,
                       const SArray<uint64_t> &push_keys,
                       const SArray<float> &data,
                       const SArray<version_t> &updates);

} // namespace ps
