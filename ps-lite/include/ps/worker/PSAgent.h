#pragma once

#include "ps/ps.h"
#include "ps/worker/kvworker.h"
#include "ps/psf/PSFunc.h"
#include "ps/server/param.h"
#include "common/logging.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <sys/time.h>
#include <unistd.h>
#include <unordered_map>
#include <set>
#include <chrono>
#include <numeric>
#include <map>
#include <mutex>

namespace ps {

struct TensorMeta {
    ParamType ptype;
    size_t length;
    size_t width = 1;
    /* split a tensor into multiple pieces. [node_name -->
     * splitted_dl_array_keys] */
    vector<Key> keys;
    /* [node_name --> timestamp to be waited] */
    std::vector<int> ts;
    std::vector<size_t> part;
};

struct SparseInfos {
    // store structures used in sparse operations to avoid memory leak
    // if using C++17 can changed to shared_ptr (which supports dynamic arrays)
    size_t *in_offset;
    size_t *out_offset;
    float *in_data;
};

/*
 * A singleton object for pulling or push to PS.
 * Since we enable sparse pull/push in PSVector and the length of each val is
 * one, thus the $lens in @kvpairs is not useful. As a result, we use $lens to
 * store the offset of each vector. for example, key=1000, lens = {1,2,3}, then
 * we are accessing elements with ids as {1000+1, 1000+2, 1000+3}
 */
class PSAgent {
private:
    /* The KVWorker used to make requests. */
    KVWorker _kvworker;
    Partitioner *_par;
    std::unordered_map<int, TensorMeta> _id2meta;
    std::unordered_map<int, SparseInfos> _id2sparseinfo;

    Key _globalId = 0;

    /* for round-robin tensor placement */
    size_t _serverIndex = 0;

    PSAgent() : _kvworker(0, 0) {
        _par = _kvworker.par;
    }

public:
    static PSAgent *Get() {
        static PSAgent e;
        return &e;
    }

    void wait(int name) {
        for (int t : _id2meta[name].ts)
            _kvworker.Wait(t);
        _id2meta[name].ts.clear();
    }

    void clear(int name) {
        _id2meta.erase(name);
        // TODO: delete on PS
    }

    void clearOnServer(int name) {
        TensorMeta &meta = _id2meta[name];
        for (size_t i = 0; i < meta.keys.size(); i++) {
            PSFData<ParamClear>::Request request(meta.keys[i]);
            auto cb = getCallBack<ParamClear>();
            meta.ts.push_back(_kvworker.Request<ParamClear>(request, cb));
        }
        wait(name);
    }

    void waitTimestamp(int timestamp) {
        _kvworker.Wait(timestamp);
    }

    /**
     * \brief init the meta information about this data on PS.
     *        the meta data is stored on each worker.
     * \param name the name of the input data
     * \param cols the #columns of the data, the data are partitioned by cols.
     */
    void registerTensor(const int name, const ParamType ptype,
                        const size_t length, const size_t width = 1) {
        assert(!_id2meta.count(name));
        TensorMeta tm;
        tm.ptype = ptype;
        tm.length = length;
        if (ptype == kParam) {
            _par->partitionDense(length, tm.keys, tm.part);
        } else {
            tm.width = width;
            _par->partitionSparse(length, width, tm.keys, tm.part);
            SparseInfos sp;
            sp.in_offset = nullptr;
            sp.out_offset = nullptr;
            sp.in_data = nullptr;
            _id2sparseinfo[name] = sp;
        }
        _id2meta[name] = tm;
    }

    void vecPushSparse(const int name, float *dup_index, float *vals,
                       const size_t dup_index_size, int priority = 0) {
        TensorMeta &meta = _id2meta[name];
        const std::vector<Key> &keys = meta.keys;
        const std::vector<size_t> &lens = meta.part;
        size_t width = meta.width;
        SparseInfos &sp = _id2sparseinfo[name];
        delete[] sp.in_offset;
        delete[] sp.in_data;

        std::map<size_t, std::vector<size_t>> idx2map;
        for (size_t i = 0; i < dup_index_size; ++i) {
            size_t idx = (size_t)dup_index[i];
            idx2map[idx].emplace_back(i);
        }

        size_t index_size = idx2map.size();
        size_t num_all = index_size * width;
        size_t *cp_offset = sp.in_offset = new size_t[index_size];
        float *cp_val = sp.in_data = new float[num_all]();

        size_t cur_index = 0;
        size_t cur_offset = 0;
        size_t cur_len = 0;
        auto iter = idx2map.begin();
        std::vector<std::pair<bool, int>> ts(keys.size());

        for (size_t i = 0; i < keys.size(); ++i) {
            size_t st_index = cur_index;
            size_t st_offset = cur_offset;
            while (iter != idx2map.end() && iter->first < cur_len + lens[i]) {
                cp_offset[cur_index++] = iter->first - cur_len;
                for (auto j : iter->second) {
                    size_t ori_offset = j * width;
                    for (size_t k = 0; k < width; ++k) {
                        cp_val[cur_offset + k] += vals[ori_offset + k];
                    }
                }
                cur_offset += width;
                ++iter;
            }
            if (cur_index > st_index) {
                ts[i].first = true;
                PSFData<SparsePush>::Request request(
                    keys[i],
                    SArray<size_t>(cp_offset + st_index, cur_index - st_index),
                    SArray<float>(cp_val + st_offset, cur_offset - st_offset));
                auto cb = getCallBack<SparsePush>();
                ts[i].second = _kvworker.Request<SparsePush>(request, cb);
            } else {
                ts[i].first = false;
            }
            cur_len += lens[i];
        }

        for (auto &t : ts)
            if (t.first)
                meta.ts.push_back(t.second);
        return;
    }

    void vecPullSparse(const int name, float *dup_index, float *vals,
                       const size_t dup_index_size, int priority = 0) {
        TensorMeta &meta = _id2meta[name];
        const std::vector<Key> &keys = meta.keys;
        const std::vector<size_t> &lens = meta.part;
        size_t width = meta.width;
        SparseInfos &sp = _id2sparseinfo[name];
        delete[] sp.out_offset;

        std::map<size_t, std::vector<size_t>> idx2map;
        for (size_t i = 0; i < dup_index_size; ++i) {
            size_t idx = (size_t)dup_index[i];
            idx2map[idx].emplace_back(i);
        }

        size_t index_size = idx2map.size();
        size_t *cp_offset = sp.out_offset = new size_t[index_size];

        size_t cur_index = 0;
        size_t cur_len = 0;
        auto iter = idx2map.begin();
        std::vector<std::pair<bool, int>> ts(keys.size());

        for (size_t i = 0; i < keys.size(); ++i) {
            size_t st_index = cur_index;
            auto st_iter = iter;
            while (iter != idx2map.end() && iter->first < cur_len + lens[i]) {
                cp_offset[cur_index++] = iter->first - cur_len;
                ++iter;
            }
            if (cur_index > st_index) {
                ts[i].first = true;
                PSFData<SparsePull>::Request request(
                    keys[i],
                    SArray<size_t>(cp_offset + st_index, cur_index - st_index));
                auto cb = getCallBack<SparsePull>(
                    SArray<float>(vals, dup_index_size * width),
                    std::move(
                        std::vector<std::pair<size_t, std::vector<size_t>>>(
                            st_iter, iter)),
                    cur_len, width);
                ts[i].second = _kvworker.Request<SparsePull>(request, cb);
            } else {
                ts[i].first = false;
            }
            cur_len += lens[i];
        }

        for (auto &t : ts)
            if (t.first)
                meta.ts.push_back(t.second);
        return;
    }

    void vecSDPushPull(const int name, float *dup_index, float *vals,
                       const size_t dup_index_size, float *out_vals,
                       int priority = 0) {
        TensorMeta &meta = _id2meta[name];
        const std::vector<Key> &keys = meta.keys;
        const std::vector<size_t> &lens = meta.part;
        size_t width = meta.width;
        SparseInfos &sp = _id2sparseinfo[name];
        delete[] sp.in_offset;
        delete[] sp.in_data;

        std::map<size_t, std::vector<size_t>> idx2map;
        for (size_t i = 0; i < dup_index_size; ++i) {
            size_t idx = (size_t)dup_index[i];
            idx2map[idx].emplace_back(i);
        }

        size_t index_size = idx2map.size();
        size_t num_all = index_size * width;
        size_t *cp_offset = sp.in_offset = new size_t[index_size];
        float *cp_val = sp.in_data = new float[num_all]();

        size_t cur_index = 0;
        size_t cur_offset = 0;
        size_t cur_len = 0;
        size_t pull_offset = 0;
        auto iter = idx2map.begin();

        for (size_t i = 0; i < keys.size(); ++i) {
            size_t st_index = cur_index;
            size_t st_offset = cur_offset;
            size_t local_length = lens[i] * width;
            while (iter != idx2map.end() && iter->first < cur_len + lens[i]) {
                cp_offset[cur_index++] = iter->first - cur_len;
                for (auto j : iter->second) {
                    size_t ori_offset = j * width;
                    for (size_t k = 0; k < width; ++k) {
                        cp_val[cur_offset + k] += vals[ori_offset + k];
                    }
                }
                cur_offset += width;
                ++iter;
            }
            PSFData<SDPushPull>::Request request(
                keys[i],
                SArray<size_t>(cp_offset + st_index, cur_index - st_index),
                SArray<float>(cp_val + st_offset, cur_offset - st_offset),
                local_length);
            auto cb = getCallBack<SDPushPull>(
                SArray<float>(out_vals + pull_offset, local_length));
            meta.ts.push_back(_kvworker.Request<SDPushPull>(request, cb));
            cur_len += lens[i];
            pull_offset += local_length;
        }
        return;
    }

    void vecSSPushPull(const int name, float *in_index, float *in_vals,
                       float *out_index, float *out_vals,
                       const size_t dup_index_size, int priority = 0) {
        TensorMeta &meta = _id2meta[name];
        const std::vector<Key> &keys = meta.keys;
        const std::vector<size_t> &lens = meta.part;
        size_t width = meta.width;
        SparseInfos &sp = _id2sparseinfo[name];
        delete[] sp.in_offset;
        delete[] sp.out_offset;
        delete[] sp.in_data;

        std::map<size_t, std::vector<size_t>> in_idx2map;
        std::map<size_t, std::vector<size_t>> out_idx2map;
        for (size_t i = 0; i < dup_index_size; ++i) {
            size_t idx = (size_t)in_index[i];
            in_idx2map[idx].emplace_back(i);
            idx = (size_t)out_index[i];
            out_idx2map[idx].emplace_back(i);
        }

        size_t in_index_size = in_idx2map.size();
        size_t out_index_size = out_idx2map.size();
        size_t in_num_all = in_index_size * width;
        size_t *in_cp_offset = sp.in_offset = new size_t[in_index_size];
        size_t *out_cp_offset = sp.out_offset = new size_t[out_index_size];
        float *in_cp_val = sp.in_data = new float[in_num_all]();

        size_t in_cur_index = 0;
        size_t in_cur_offset = 0;
        size_t cur_len = 0;
        size_t out_cur_index = 0;
        auto in_iter = in_idx2map.begin();
        auto out_iter = out_idx2map.begin();
        std::vector<std::pair<bool, int>> ts(keys.size());

        for (size_t i = 0; i < keys.size(); ++i) {
            size_t in_st_index = in_cur_index;
            size_t st_offset = in_cur_offset;
            while (in_iter != in_idx2map.end()
                   && in_iter->first < cur_len + lens[i]) {
                in_cp_offset[in_cur_index++] = in_iter->first - cur_len;
                for (auto j : in_iter->second) {
                    size_t ori_offset = j * width;
                    for (size_t k = 0; k < width; ++k) {
                        in_cp_val[in_cur_offset + k] += in_vals[ori_offset + k];
                    }
                }
                in_cur_offset += width;
                ++in_iter;
            }

            size_t out_st_index = out_cur_index;
            auto st_iter = out_iter;
            while (out_iter != out_idx2map.end()
                   && out_iter->first < cur_len + lens[i]) {
                out_cp_offset[out_cur_index++] = out_iter->first - cur_len;
                ++out_iter;
            }

            if (in_cur_index > in_st_index || out_cur_index > out_st_index) {
                ts[i].first = true;
                PSFData<SSPushPull>::Request request(
                    keys[i],
                    SArray<size_t>(in_cp_offset + in_st_index,
                                   in_cur_index - in_st_index),
                    SArray<float>(in_cp_val + st_offset,
                                  in_cur_offset - st_offset),
                    SArray<size_t>(out_cp_offset + out_st_index,
                                   out_cur_index - out_st_index));
                auto cb = getCallBack<SparsePull>(
                    SArray<float>(out_vals, dup_index_size * width),
                    std::move(
                        std::vector<std::pair<size_t, std::vector<size_t>>>(
                            st_iter, out_iter)),
                    cur_len, width);
                ts[i].second = _kvworker.Request<SSPushPull>(request, cb);
            } else {
                ts[i].first = false;
            }
            cur_len += lens[i];
        }

        for (auto &t : ts)
            if (t.first)
                meta.ts.push_back(t.second);
        return;
    }

    /**
     * \brief PSVector: pull <Key, floats> pairs from PS.
     * \param name name of the PSVector
     * \param vals the vals of pullsh vals
     */
    void vecDensePush(const int name, float *vals, int priority = 0) {
        TensorMeta &meta = _id2meta[name];
        auto cb = getCallBack<DensePush>();
        /* send push request to each partition according to the offsets. */
        size_t cur_len = 0;
        for (size_t i = 0; i < meta.keys.size(); i++) {
            PSFData<DensePush>::Request request(
                meta.keys[i], meta.part[i],
                SArray<float>(vals + cur_len, meta.part[i]));
            meta.ts.push_back(_kvworker.Request<DensePush>(request, cb));
            cur_len += meta.part[i];
        }
    }

    void vecDensePull(const int name, float *vals, int priority = 0) {
        TensorMeta &meta = _id2meta[name];
        size_t cur_offset = 0;
        for (size_t i = 0; i < meta.keys.size(); i++) {
            size_t cur_length = meta.part[i] * meta.width;
            PSFData<DensePull>::Request request(meta.keys[i], cur_length);
            auto cb = getCallBack<DensePull>(
                SArray<float>(vals + cur_offset, cur_length));
            meta.ts.push_back(_kvworker.Request<DensePull>(request, cb));
            cur_offset += cur_length;
        }
    }

    void vecDDPushPull(const int name, float *in_vals, float *out_vals,
                       int priority = 0) {
        TensorMeta &meta = _id2meta[name];
        size_t cur_len = 0;
        /* send pull request to each partition */
        for (size_t i = 0; i < meta.keys.size(); i++) {
            PSFData<DDPushPull>::Request request(
                meta.keys[i], meta.part[i],
                SArray<float>(in_vals + cur_len, meta.part[i]));
            auto cb = getCallBack<DDPushPull>(
                SArray<float>(out_vals + cur_len, meta.part[i]));
            meta.ts.push_back(_kvworker.Request<DDPushPull>(request, cb));
            cur_len += meta.part[i];
        }
    }

    void ParameterInit(const int name, InitType init_type, double init_a,
                       double init_b, unsigned long long seed, OptType otype,
                       SArray<float> lrs) {
        TensorMeta &meta = _id2meta[name];
        /* send pull request to each partition */
        auto cb = getCallBack<ParamInit>();
        for (size_t i = 0; i < meta.keys.size(); i++) {
            PSFData<ParamInit>::Request request(
                meta.keys[i], meta.ptype, meta.part[i], meta.width, init_type,
                init_a, init_b, seed, otype, lrs);
            meta.ts.push_back(_kvworker.Request<ParamInit>(request, cb));
        }
    }

    void ParameterSave(const int name, char *address) {
        TensorMeta &meta = _id2meta[name];
        /* send pull request to each partition */
        auto cb = getCallBack<ParamSave>();
        for (size_t i = 0; i < meta.keys.size(); i++) {
            std::string local_address = std::string(address) + "/"
                                        + std::to_string(name) + "_"
                                        + std::to_string(i) + ".dat";
            SArray<char> temp_array;
            temp_array.CopyFrom(local_address.c_str(), local_address.size());
            PSFData<ParamSave>::Request request(meta.keys[i], temp_array,
                                                false);
            meta.ts.push_back(_kvworker.Request<ParamSave>(request, cb));
        }
    }

    void ParameterLoad(const int name, char *address) {
        TensorMeta &meta = _id2meta[name];
        /* send pull request to each partition */
        auto cb = getCallBack<ParamLoad>();
        for (size_t i = 0; i < meta.keys.size(); i++) {
            std::string local_address = std::string(address) + "/"
                                        + std::to_string(name) + "_"
                                        + std::to_string(i) + ".dat";
            SArray<char> temp_array;
            temp_array.CopyFrom(local_address.c_str(), local_address.size());
            PSFData<ParamLoad>::Request request(meta.keys[i], temp_array);
            meta.ts.push_back(_kvworker.Request<ParamLoad>(request, cb));
        }
    }

    void startRecord(std::string dirPath) {
        _kvworker.startRecord(dirPath);
    }

    void getLoads() {
        _kvworker.recordLoads();
    }

    void SSPSync(Key key, ssp_version_t version) {
        PSFData<kSSPSync>::Request request(key, Postoffice::Get()->my_rank(), version);
        bool success = false;
        auto cb = getCallBack<kSSPSync>(std::ref(success));
        while (!success) {
            _kvworker.Wait(_kvworker.Request<kSSPSync>(request, cb));
        }
    }

    void SSPInit(Key key, size_t group_size, ssp_version_t tolerance) {
        PSFData<kSSPInit>::Request request(key, Postoffice::Get()->my_rank(), group_size, tolerance);
        auto cb = getCallBack<kSSPInit>();
        _kvworker.Wait(_kvworker.Request<kSSPInit>(request, cb));
    }

    void PReduceGetPartner(Key key, int rank, size_t required_worker_num, float wait_time, int* result) {
        PSFData<kPReduceGetPartner>::Request request(key, rank, required_worker_num, wait_time);
        auto cb = getCallBack<kPReduceGetPartner>(result);
        _kvworker.Wait(_kvworker.Request<kPReduceGetPartner>(request, cb));
    }

    /*
        A simple key mapping for multiple server case
    */
    Key mapWkeyToSkey(Key idx) {
        const std::vector<Range> &server_range =
            Postoffice::Get()->GetServerKeyRanges();
        int server = idx % server_range.size();
        Key k = server_range[server].end() - idx - 1;
        return k;
    }

    /*
        Enqueue the Zpush request for PushData
    */
    void PushData(Key idx, float *vals, int len, std::vector<int> &timestamp) {
        auto cb = getCallBack<DensePush>();
        PSFData<DensePush>::Request request(mapWkeyToSkey(idx), len,
                                            SArray<float>(vals, len));
        int ts = _kvworker.Request<DensePush>(request, cb);
        timestamp.push_back(ts);
    }

    // This is almost the same as PushData
    void PullData(Key idx, float *vals, int len, std::vector<int> &timestamp) {
        auto cb = getCallBack<DensePull>(SArray<float>(vals, len));
        PSFData<DensePull>::Request request(mapWkeyToSkey(idx), len);
        int ts = _kvworker.Request<DensePull>(request, cb);
        timestamp.push_back(ts);
    }

    void syncEmbedding(int name, const SArray<uint64_t> &rows,
                       const SArray<version_t> &ver, version_t bound,
                       PSFData<kSyncEmbedding>::Closure closure) {
        TensorMeta &meta = _id2meta[name];
        size_t start = 0, end = 0, cur_len = 0;
        for (size_t i = 0; i < meta.keys.size(); i++) {
            // find the idx range
            start = end;
            end = std::lower_bound(rows.begin() + start, rows.end(),
                                   cur_len + meta.part[i])
                  - rows.begin();
            if (start == end)
                continue;
            // remove row offset inplace so that index fits with server
            SArray<uint64_t> new_rows = rows.segment(start, end);
            for (size_t i = 0; i < new_rows.size(); i++)
                new_rows[i] -= cur_len;
            PSFData<kSyncEmbedding>::Request request(
                meta.keys[i], new_rows, ver.segment(start, end), bound);
            auto cb = std::bind(closure, std::placeholders::_1, start);
            meta.ts.push_back(_kvworker.Request<kSyncEmbedding>(request, cb));
            cur_len += meta.part[i];
        }
    }

    void pushEmbedding(int name, const SArray<uint64_t> &rows,
                       const SArray<float> &data,
                       const SArray<version_t> &updates) {
        TensorMeta &meta = _id2meta[name];
        size_t start = 0, end = 0, cur_len = 0;
        auto cb = getCallBack<kPushEmbedding>();
        for (size_t i = 0; i < meta.keys.size(); i++) {
            // find the idx range
            start = end;
            end = std::lower_bound(rows.begin() + start, rows.end(),
                                   cur_len + meta.part[i])
                  - rows.begin();
            if (start == end)
                continue;
            // remove row offset inplace so that index fits with server
            SArray<uint64_t> new_rows = rows.segment(start, end);
            for (size_t i = 0; i < new_rows.size(); i++)
                new_rows[i] -= cur_len;
            PSFData<kPushEmbedding>::Request request(
                meta.keys[i], new_rows,
                data.segment(start * meta.width, end * meta.width),
                updates.segment(start, end));
            meta.ts.push_back(_kvworker.Request<kPushEmbedding>(request, cb));
            cur_len += meta.part[i];
        }
    }

    void pushSyncEmbedding(int name, const SArray<uint64_t> &rows,
                           const SArray<version_t> &ver, version_t bound,
                           PSFData<kSyncEmbedding>::Closure closure,
                           const SArray<uint64_t> &push_rows,
                           const SArray<float> &data,
                           const SArray<version_t> &updates) {
        TensorMeta &meta = _id2meta[name];
        size_t start = 0, end = 0, cur_len = 0, push_start = 0, push_end = 0;
        for (size_t i = 0; i < meta.keys.size(); i++) {
            // find the idx range
            start = end;
            push_start = push_end;
            end = std::lower_bound(rows.begin() + start, rows.end(),
                                   cur_len + meta.part[i])
                  - rows.begin();
            push_end = std::lower_bound(push_rows.begin() + push_start,
                                        push_rows.end(), cur_len + meta.part[i])
                       - push_rows.begin();
            if (start == end && push_start == push_end)
                continue;
            // remove row offset inplace so that index fits with server
            SArray<uint64_t> new_rows = rows.segment(start, end),
                             new_push_rows =
                                 push_rows.segment(push_start, push_end);
            for (size_t i = 0; i < new_rows.size(); i++)
                new_rows[i] -= cur_len;
            for (size_t i = 0; i < new_push_rows.size(); i++)
                new_push_rows[i] -= cur_len;
            PSFData<kPushSyncEmbedding>::Request request(
                meta.keys[i], new_rows, ver.segment(start, end), bound,
                new_push_rows,
                data.segment(push_start * meta.width, push_end * meta.width),
                updates.segment(push_start, push_end));
            auto cb = std::bind(closure, std::placeholders::_1, start);
            meta.ts.push_back(
                _kvworker.Request<kPushSyncEmbedding>(request, cb));
            cur_len += meta.part[i];
        }
    }
};

} // namespace ps
