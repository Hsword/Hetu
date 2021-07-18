#pragma once

#include <unordered_map>
#include <memory>
#include <sstream>
#include "binding.h"

using std::default_delete;
using std::make_shared;
using std::shared_ptr;
using std::unordered_map;

namespace hetu {

typedef uint64_t cache_key_t;
typedef int64_t version_t;

template <typename T>
class Line {
private:
    version_t version_;
    version_t updates_;
    const size_t len_;
    const cache_key_t key_;
    T *data_;
    T *grad_;

public:
    typedef T dtype;
    Line(cache_key_t key, const T *in_vec, size_t len) : len_(len), key_(key) {
        data_ = new T[len]();
        grad_ = nullptr;
        std::copy(in_vec, in_vec + len, data_);
        updates_ = 0;
        version_ = -1;
    }
    Line(cache_key_t key, size_t len, bool init_data = true) :
        len_(len), key_(key) {
        data_ = init_data ? new T[len]() : nullptr;
        grad_ = nullptr;
        updates_ = 0;
        version_ = -1;
    }
    Line(const Line &other) = delete;
    ~Line() {
        delete[] data_;
        delete[] grad_;
    }
    //-------------------------- setter getter ---------------------------------
    T &operator[](size_t i) {
        return data_[i];
    }
    T *data() {
        return data_;
    }
    T *grad() {
        _maybeInitGrad();
        return grad_;
    }
    size_t size() {
        return len_;
    }
    cache_key_t key() const {
        return key_;
    }

    void setVersion(version_t version) {
        version_ = version;
    }
    version_t getVersion() {
        return version_;
    }
    version_t getUpdates() {
        return updates_;
    }

    //-------------------------- handling gradients ----------------------------
    void accumulate(const T *in_grad) {
        _maybeInitGrad();
        if (!data_) {
            for (size_t i = 0; i < len_; i++) {
                grad_[i] += in_grad[i];
            }
        } else {
            for (size_t i = 0; i < len_; i++) {
                grad_[i] += in_grad[i];
                data_[i] += in_grad[i];
            }
        }
        updates_++;
    }
    void addup() {
        if (grad_)
            for (size_t i = 0; i < len_; i++)
                data_[i] += grad_[i];
    }
    double mean() const {
        double sum = 0;
        for (size_t i = 0; i < len_; i++) {
            sum += data_[i];
        }
        return sum / len_;
    }
    double var() const {
        double avg = mean();
        double sum = 0;
        for (size_t i = 0; i < len_; i++) {
            sum += (data_[i] - avg) * (data_[i] - avg);
        }
        return sum / len_;
    }
    void zeroGrad() {
        _maybeInitGrad();
        for (size_t i = 0; i < len_; i++) {
            grad_[i] = 0;
        }
        updates_ = 0;
    }

    void _maybeInitGrad() {
        if (grad_ == nullptr)
            grad_ = new T[len_]();
    }

    //------------------------ python api starts here ------------------------
    // __repr__ is used in python embedding
    std::string __repr__() {
        std::stringstream ss;
        ss << "<hetu.Embedding ";
        ss << ": "
           << "key:" << key_;
        ss << ", "
           << "len:" << len_;
        ss << ", "
           << "version:" << version_;
        ss << ", "
           << "mean:" << mean();
        ss << ", "
           << "var:" << var();
        ss << ">";
        return ss.str();
    }
    py::array_t<T> PyAPI_data() {
        return bind::pt1d_nocp(data_, len_);
    }
    py::array_t<T> PyAPI_grad() {
        return bind::pt1d_nocp(grad_, len_);
    }
}; // class Line

typedef float embed_t;
// Embedding is a float embedding table, usually 128 length
typedef Line<embed_t> Embedding;
// EmbeddingPT is the smart pointer type of Embedding
typedef shared_ptr<Embedding> EmbeddingPT;

// Factory function
EmbeddingPT makeEmbedding(cache_key_t, version_t, py::array_t<embed_t>);

} // namespace hetu
