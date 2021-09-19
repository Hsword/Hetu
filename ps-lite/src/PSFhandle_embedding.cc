#include "ps/server/PSFHandle.h"

namespace ps {

void PSHandler<PsfGroup::kParameterServer>::serve(
    const PSFData<kPushEmbedding>::Request &request,
    PSFData<kPushEmbedding>::Response &response) {
    Key k = get<0>(request);
    auto rows = get<1>(request);
    auto data = get<2>(request);
    auto updates = get<3>(request);
    auto iter = const_store.find(k);
    CHECK_NE(iter, const_store.end()) << "key does not exist";
    CHECK_EQ(iter->second->type(), kCacheTable) << " key is not Cachetable";
    auto &value_set =
        *std::dynamic_pointer_cast<CacheTable<float>>(iter->second);
    size_t width = value_set.width;
    CHECK_EQ(updates.size(), rows.size())
        << "PushEmbedding updates size mismatch";
    CHECK_EQ(data.size(), rows.size() * width)
        << "PushEmbedding data size mismatch";
    auto write_lock = value_set.write_guard();
    for (size_t i = 0; i < rows.size(); i++) {
        value_set.ver[rows[i]] += updates[i];
        for (size_t j = 0; j < width; j++)
            value_set[rows[i] * width + j] += data[i * width + j];
    }
}

void PSHandler<PsfGroup::kParameterServer>::serve(
    const PSFData<kSyncEmbedding>::Request &request,
    PSFData<kSyncEmbedding>::Response &response) {
    Key k = get<0>(request);
    auto rows = get<1>(request);
    auto ver = get<2>(request);
    auto bound = get<3>(request);
    auto &idx = get<0>(response);
    auto &ret_ver = get<1>(response);
    auto &data = get<2>(response);
    auto iter = const_store.find(k);
    CHECK_NE(iter, const_store.end()) << "key does not exist";
    CHECK_EQ(iter->second->type(), kCacheTable) << " key is not Cachetable";
    auto &value_set =
        *std::dynamic_pointer_cast<CacheTable<float>>(iter->second);
    size_t width = value_set.width;
    auto read_lock = value_set.read_guard();
    size_t count = 0;
    for (size_t i = 0; i < rows.size(); i++)
        if (ver[i] == -1 || value_set.ver[rows[i]] - ver[i] > bound)
            count++;
    idx.resize(count);
    ret_ver.resize(count);
    data.resize(count * width);
    count = 0;
    for (size_t i = 0; i < rows.size(); i++) {
        if (ver[i] == -1 || value_set.ver[rows[i]] - ver[i] > bound) {
            idx[count] = i;
            ret_ver[count] = value_set.ver[rows[i]];
            std::copy(&value_set[rows[i] * width],
                      &value_set[(rows[i] + 1) * width], &data[count * width]);
            count++;
        }
    }
}

void PSHandler<PsfGroup::kParameterServer>::serve(
    const PSFData<kPushSyncEmbedding>::Request &request,
    PSFData<kPushSyncEmbedding>::Response &response) {
    PSFData<kPushEmbedding>::Request push_req(
        std::get<0>(request), std::get<4>(request), std::get<5>(request),
        std::get<6>(request));
    PSFData<kPushEmbedding>::Response push_res;
    serve(push_req, push_res);

    PSFData<kSyncEmbedding>::Request sync_req(
        std::get<0>(request), std::get<1>(request), std::get<2>(request),
        std::get<3>(request));
    serve(sync_req, response);
}

} // namespace ps
