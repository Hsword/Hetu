#pragma once

#include "ps/psf/PSFunc.h"

#include "common/thread_safe_hash_map.h"
#include "param.h"
#include <algorithm>
#include <utility>
#include <mutex>
#include <omp.h>
#include <random>
#include <fstream>

namespace ps {

template <>
class PSHandler<PsfGroup::kParameterServer>
    : public PSHandler<PsfGroup::kBaseGroup> {
public:
    PSHandler<PsfGroup::kParameterServer>() {
    }
    PSHandler<PsfGroup::kParameterServer>(
        const PSHandler<PsfGroup::kParameterServer> &handle) {
    }

    void serve(const PSFData<DensePull>::Request &request,
               PSFData<DensePull>::Response &response) {
        Key k = get<0>(request);
        size_t len = get<1>(request);
        SArray<float> &pull_vals = get<0>(response);

        auto iter = const_store.find(k);
        if (iter != const_store.end()) {
            auto &value_set_ = *iter->second;
            size_t data_size = value_set_.size();
            CHECK_EQ(len, data_size) << " size mismatch in DensePull " << k
                                     << " " << len << " " << data_size;
            pull_vals.resize(data_size);
            auto read_lock = value_set_.read_guard();
            std::copy(value_set_.begin(), value_set_.end(), pull_vals.begin());
        } else {
            LG << "Key does not exist on PS in DensePull" << k;
        }
    }

    void serve(const PSFData<DensePush>::Request &request,
               PSFData<DensePush>::Response &response) {
        Key k = get<0>(request);
        size_t len = get<1>(request);
        SArray<float> vals = get<2>(request);

        if (const_store.find(k) == const_store.end()) {
            store[k] = std::make_shared<Param<float>>(len, OptType::None,
                                                      SArray<float>());
        }
        auto iter = const_store.find(k);
        if (iter != const_store.end()) {
            CHECK_EQ(len, iter->second->size())
                << k << " " << len << " " << iter->second->size()
                << " size mismatch in DensePush";
            // write, discard const qualifier
            auto &value_set_ =
                *const_cast<typename tmap::mapped_type &>(iter->second);
            auto write_lock = value_set_.write_guard();
#pragma omp parallel for num_threads(4)
            for (size_t j = 0; j < value_set_.size(); j++)
                value_set_[j] += vals[j];
        } else {
            LG << "Key does not exist on PS in DensePull" << k;
        }
    }

    void serve(const PSFData<DDPushPull>::Request &request,
               PSFData<DDPushPull>::Response &response) {
        // one key per request.
        // with response result
        Key k = get<0>(request);
        size_t len = get<1>(request);
        SArray<float> vals = get<2>(request);
        SArray<float> &pull_vals = get<0>(response);

        auto iter = const_store.find(k);
        if (iter != const_store.end()) {
            auto &value_set_ =
                *const_cast<typename tmap::mapped_type &>(iter->second);
            size_t data_size = value_set_.size();
            CHECK_EQ(len, data_size)
                << " size mismatch in DDPushPull " << len << " " << data_size;
            pull_vals.resize(data_size);
            auto write_lock = value_set_.write_guard();
#pragma omp parallel for num_threads(4)
            for (size_t j = 0; j < data_size; j++) {
                value_set_[j] += vals[j];
                pull_vals[j] = value_set_[j];
            }
        } else {
            LG << "Key does not exist on PS in DensePull" << k;
        }
    }

    void serve(const PSFData<SparsePull>::Request &request,
               PSFData<SparsePull>::Response &response) {
        // we use length as the offset, i.e., #length = #vals.
        // with response result
        Key k = get<0>(request);
        SArray<size_t> offset = get<1>(request);
        SArray<float> &pull_vals = get<0>(response);

        auto iter = const_store.find(k);
        if (iter != const_store.end()) {
            auto &value_set_ =
                *std::dynamic_pointer_cast<Param2D<float>>(iter->second);
            size_t width = value_set_.width;
            pull_vals.resize(offset.size() * width);
            auto read_lock = value_set_.read_guard();
#pragma omp parallel for num_threads(4)
            for (size_t j = 0; j < offset.size(); ++j) {
                auto value_begin = value_set_.data() + offset[j] * width;
                auto value_end = value_begin + width;
                auto dst_begin = pull_vals.data() + j * width;
                std::copy(value_begin, value_end, dst_begin);
            }
        } else {
            // error, the key does not exist on PS.
            LF << "[Error] The pulled key: " << k
               << " does not exist on PS in SparsePull.";
        }
    }

    void serve(const PSFData<SparsePush>::Request &request,
               PSFData<SparsePush>::Response &response) {
        // we use length as the offset, i.e., #length = #vals.
        // no response result
        Key k = get<0>(request);
        SArray<size_t> offsets = get<1>(request);
        SArray<float> vals = get<2>(request);

        auto iter = const_store.find(k);
        if (iter != const_store.end()) {
            auto &value_set_ =
                *std::dynamic_pointer_cast<Param2D<float>>(iter->second);
            size_t width = value_set_.width;

            CHECK_EQ(vals.size(), offsets.size() * width)
                << " in Psf::SparsePush check failed,"
                << " size of vals is " << vals.size() << " size of lens is "
                << offsets.size() << " size of width is " << width;

            // write, discard const qualifier
            auto write_lock = value_set_.write_guard();
#pragma omp parallel for num_threads(4)
            for (size_t j = 0; j < offsets.size(); ++j) {
                size_t src_offset = j * width;
                size_t dst_offset = offsets[j] * width;
                for (size_t k = 0; k < width; ++k) {
                    value_set_[dst_offset + k] += vals[src_offset + k];
                }
            }
        } else {
            // error, the key does not exist on PS.
            LF << "[Error] The pushed key: " << k
               << " does not exist on PS in SparsePush.";
        }
    }

    void serve(const PSFData<SDPushPull>::Request &request,
               PSFData<SDPushPull>::Response &response) {
        Key k = get<0>(request);
        SArray<size_t> offsets = get<1>(request);
        SArray<float> vals = get<2>(request);
        size_t len = get<3>(request);
        SArray<float> &pull_vals = get<0>(response);

        auto iter = const_store.find(k);
        if (iter != const_store.end()) {
            auto &value_set_ =
                *std::dynamic_pointer_cast<Param2D<float>>(iter->second);
            size_t width = value_set_.width;
            CHECK_EQ(len, value_set_.size())
                << " size mismatch in SDPushPull " << k << " " << len << " "
                << value_set_.size();

            // sparsepush phase
            if (vals.size() > 0) {
                CHECK_EQ(vals.size(), offsets.size() * width)
                    << " in Psf::SDPushPull check failed,"
                    << " size of vals is " << vals.size() << " size of lens is "
                    << offsets.size() << " size of width is " << width;

                // write, discard const qualifier
                auto write_lock = value_set_.write_guard();
#pragma omp parallel for num_threads(4)
                for (size_t j = 0; j < offsets.size(); ++j) {
                    size_t src_offset = j * width;
                    size_t dst_offset = offsets[j] * width;
                    for (size_t k = 0; k < width; ++k) {
                        value_set_[dst_offset + k] += vals[src_offset + k];
                    }
                }
            }
            // densepull phase
            pull_vals.resize(value_set_.size());
            auto read_lock = value_set_.read_guard();
            std::copy(value_set_.begin(), value_set_.end(), pull_vals.begin());
        } else {
            // error, the key does not exist on PS.
            LF << "[Error] The pushed key: " << k
               << " does not exist on PS in SDPushPull.";
        }
    }

    void serve(const PSFData<SSPushPull>::Request &request,
               PSFData<SSPushPull>::Response &response) {
        Key k = get<0>(request);
        SArray<size_t> push_offsets = get<1>(request);
        SArray<float> vals = get<2>(request);
        SArray<size_t> pull_offsets = get<3>(request);
        SArray<float> &pull_vals = get<0>(response);

        auto iter = const_store.find(k);
        if (iter != const_store.end()) {
            auto &value_set_ =
                *std::dynamic_pointer_cast<Param2D<float>>(iter->second);
            size_t width = value_set_.width;

            // sparsepush phase
            if (vals.size() > 0) {
                CHECK_EQ(vals.size(), push_offsets.size() * width)
                    << " in Psf::SSPushPull check failed,"
                    << " size of vals is " << vals.size() << " size of lens is "
                    << push_offsets.size() << " size of width is " << width;

                // write, discard const qualifier
                auto write_lock = value_set_.write_guard();
#pragma omp parallel for num_threads(4)
                for (size_t j = 0; j < push_offsets.size(); ++j) {
                    size_t src_offset = j * width;
                    size_t dst_offset = push_offsets[j] * width;
                    for (size_t k = 0; k < width; ++k) {
                        value_set_[dst_offset + k] += vals[src_offset + k];
                    }
                }
            }

            // sparsepull phase
            if (pull_offsets.size() > 0) {
                pull_vals.resize(pull_offsets.size() * width);
                auto read_lock = value_set_.read_guard();
#pragma omp parallel for num_threads(4)
                for (size_t j = 0; j < pull_offsets.size(); ++j) {
                    auto val_begin =
                        value_set_.begin() + pull_offsets[j] * width;
                    auto val_end = val_begin + width;
                    auto dst_begin = pull_vals.begin() + j * width;
                    std::copy(val_begin, val_end, dst_begin);
                }
            }
        } else {
            // error, the key does not exist on PS.
            LF << "[Error] The pushed key: " << k
               << " does not exist on PS in SparsePush.";
        }
    }

    void serve(const PSFData<kSyncEmbedding>::Request &request,
               PSFData<kSyncEmbedding>::Response &response);
    void serve(const PSFData<kPushEmbedding>::Request &request,
               PSFData<kPushEmbedding>::Response &response);
    void serve(const PSFData<kPushSyncEmbedding>::Request &request,
               PSFData<kPushSyncEmbedding>::Response &response);

    void serve(const PSFData<ParamInit>::Request &request,
               PSFData<ParamInit>::Response &response) {
        // one key per request.
        // no response result
        Key k = get<0>(request);
        ParamType param_type = (ParamType)get<1>(request);
        size_t len = get<2>(request);
        size_t width = get<3>(request);
        InitType init_type = (InitType)get<4>(request);
        double init_a = get<5>(request);
        double init_b = get<6>(request);
        unsigned long long seed = get<7>(request);
        OptType otype = (OptType)get<8>(request);
        SArray<float> lrs = get<9>(request);

        if (!try_init_with_no_conflict(k))
            return;

        Param<float> *newParam = nullptr;
        switch (param_type) {
        case kParam:
            newParam = new Param<float>(len, otype, lrs);
            break;
        case kParam2D:
            newParam = new Param2D<float>(len, width, otype, lrs);
            break;
        case kCacheTable:
            newParam = new CacheTable<float>(len, width, otype, lrs);
        }
        auto iter = store.find(k);
        iter->second = tmap::mapped_type(newParam);

        CHECK_EQ(len * width, iter->second->size())
            << k << " " << len << " " << width << " " << iter->second->size()
            << " size mismatch in UniformInit";
        // write, discard const qualifier
        auto &value_set_ =
            *const_cast<typename tmap::mapped_type &>(iter->second);
        auto write_lock = value_set_.write_guard();
        size_t n_threads = (value_set_.size() >> 25) + 1;
        if (n_threads > 16)
            n_threads = 16;
        if (init_type == InitType::Constant) {
            float filled_value = static_cast<float>(init_a);
            // #pragma omp parallel for num_threads(4)
            for (size_t j = 0; j < value_set_.size(); j++)
                value_set_[j] = filled_value;
        } else if (init_type == InitType::Uniform) {
            std::uniform_real_distribution<float> uniform_dist(init_a, init_b);
#pragma omp parallel num_threads(n_threads)
            {
                size_t rank = omp_get_thread_num();
                size_t num_threads = omp_get_num_threads();
                std::default_random_engine generator(seed + rank);
                size_t length = value_set_.size() / num_threads;
                size_t start = rank * length;
                size_t ending = start + length;
                if (rank == num_threads - 1)
                    ending = value_set_.size();
                for (size_t j = start; j < ending; ++j) {
                    value_set_[j] = uniform_dist(generator);
                }
            }
        } else if (init_type == InitType::Normal) {
            std::normal_distribution<float> normal_dist(init_a, init_b);
#pragma omp parallel num_threads(n_threads)
            {
                size_t rank = omp_get_thread_num();
                size_t num_threads = omp_get_num_threads();
                std::default_random_engine generator(seed + rank);
                size_t length = value_set_.size() / num_threads;
                size_t start = rank * length;
                size_t ending = start + length;
                if (rank == num_threads - 1)
                    ending = value_set_.size();
                for (size_t j = start; j < ending; ++j) {
                    value_set_[j] = normal_dist(generator);
                }
            }
        } else if (init_type == InitType::TruncatedNormal) {
            std::normal_distribution<float> truncated_normal_dist(init_a,
                                                                  init_b);
            float upper_limit = init_a + 2 * init_b;
            float lower_limit = init_a - 2 * init_b;
#pragma omp parallel num_threads(n_threads)
            {
                size_t rank = omp_get_thread_num();
                size_t num_threads = omp_get_num_threads();
                std::default_random_engine generator(seed + rank);
                size_t length = value_set_.size() / num_threads;
                size_t start = rank * length;
                size_t ending = start + length;
                if (rank == num_threads - 1)
                    ending = value_set_.size();
                for (size_t j = start; j < ending; ++j) {
                    float temp = truncated_normal_dist(generator);
                    while (temp > upper_limit || temp < lower_limit)
                        temp = truncated_normal_dist(generator);
                    value_set_[j] = temp;
                }
            }
        }
    }

    void serve(const PSFData<ParamClear>::Request &request,
               PSFData<ParamClear>::Response &response) {
        Key k = get<0>(request);
        auto iter = store.find(k);
        if (iter != store.end()) {
            store.erase(iter);
        } else {
            // error, the key does not exist on PS.
            LF << "[Error] The pushed key: " << k
               << " does not exist on PS in ParamClear.";
        }
    }

    void serve(const PSFData<ParamSave>::Request &request,
               PSFData<ParamSave>::Response &response) {
        Key k = get<0>(request);
        SArray<char> address = get<1>(request);
        auto iter = store.find(k);
        if (iter != store.end()) {
            auto &value_set_ = *iter->second;
            auto read_lock = value_set_.read_guard();
            std::ofstream fout(
                std::string(address.data(), address.size()).c_str(),
                std::ios::binary);
            fout.write((char *)value_set_.data(),
                       value_set_.size() * sizeof(float));
        } else {
            // error, the key does not exist on PS.
            LF << "[Error] The pushed key: " << k
               << " does not exist on PS in ParamSave.";
        }
    }

    void serve(const PSFData<ParamLoad>::Request &request,
               PSFData<ParamLoad>::Response &response) {
        Key k = get<0>(request);
        SArray<char> address = get<1>(request);
        auto iter = store.find(k);
        if (iter != store.end()) {
            auto &value_set_ = *iter->second;
            auto write_lock = value_set_.write_guard();
            std::ifstream fin(
                std::string(address.data(), address.size()).c_str(),
                std::ios::binary);
            fin.read((char *)value_set_.data(),
                     value_set_.size() * sizeof(float));
        } else {
            // error, the key does not exist on PS.
            LF << "[Error] The pushed key: " << k
               << " does not exist on PS in ParamLoad.";
        }
    }

private:
    bool try_init_with_no_conflict(Key key) {
        static std::mutex init_mtx;
        std::lock_guard<std::mutex> lock(init_mtx);
        if (store.find(key) != store.end())
            return false;
        else {
            store[key] = tmap::mapped_type();
            return true;
        }
    }

    typedef threadsafe_unordered_map<Key, std::shared_ptr<Param<float>>> tmap;
    tmap store;
    const tmap &const_store =
        store; // const reference to force compiler to use read lock
};

} // namespace ps
