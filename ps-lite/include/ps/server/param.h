#pragma once

#include <vector>

#include "common/shared_mutex.h"
#include "ps/psf/PSFunc.h"
#include "ps/server/optimizer.h"

namespace ps {

enum ParamType {
    kParam,
    kParam2D,
    kCacheTable,
};

/*
  Param with a read-write lock
*/
template <typename V>
class Param {
public:
    explicit Param(size_t size, OptType otype, SArray<float> lrs) {
        vec_ = new V[size]();
        size_ = size;
        switch (otype) {
        case SGD:
            opt = new SGDOptimizer<V>(lrs[0]);
            break;
        case Momentum:
            opt = new MomentumOptimizer<V>(lrs[0], lrs[1]);
            break;
        case NesterovMomentum:
            opt = new NesterovMomentumOptimizer<V>(lrs[0], lrs[1]);
            break;
        case AdaGrad:
            opt = new AdaGradOptimizer<V>(lrs[0], lrs[1], lrs[2]);
            break;
        case Adam:
            opt = new AdamOptimizer<V>(lrs[0], lrs[1], lrs[2], lrs[3]);
            break;
        case None:
            opt = nullptr;
            return;
        }
        opt->InitStates(size);
    }

    ~Param() {
        delete[] vec_;
    }

    Param(const Param &) = delete;

    s_lock<4> read_guard() const noexcept {
        return s_lock<4>(mtx);
    }
    x_lock<4> write_guard() noexcept {
        return x_lock<4>(mtx);
    }

    inline const V *data() const {
        return vec_;
    }
    inline V *data() {
        return vec_;
    }
    inline V *begin() {
        return data();
    }
    inline V *end() {
        return data() + size();
    }
    inline V &operator[](size_t i) {
        return vec_[i];
    }
    inline const V &operator[](size_t i) const {
        return vec_[i];
    }
    inline size_t size() const {
        return size_;
    }
    virtual ParamType type() {
        return kParam;
    }
    void updateDense(SArray<V> &grads) {
        auto write_lock = write_guard();
        opt->ApplyDense(*this, grads);
    }

private:
    mutable shared_mutex<4> mtx;
    V *vec_;
    size_t size_;

protected:
    Optimizer<V> *opt;
};

template <typename V>
class Param2D : public Param<V> {
public:
    explicit Param2D(size_t len, size_t wid, OptType otype, SArray<float> lrs) :
        Param<V>(len * wid, otype, lrs) {
        length = len;
        width = wid;
    }
    void updateSparse(SArray<size_t> &offsets, SArray<V> &grads) {
        auto write_lock = this->write_guard();
        this->opt->ApplySparse(*this, offsets, grads);
    }
    ParamType type() {
        return kParam2D;
    }
    size_t length, width;
};

template <typename V>
class CacheTable : public Param2D<V> {
public:
    explicit CacheTable(size_t len, size_t wid, OptType otype,
                        SArray<float> lrs) :
        Param2D<V>(len, wid, otype, lrs) {
        ver = new version_t[len]();
    }
    ~CacheTable() {
        delete[] ver;
    }
    void updateCache(SArray<version_t> &updates, SArray<size_t> &offsets,
                     SArray<V> &grads) {
        auto write_lock = this->write_guard();
        this->opt->ApplyCache(*this, updates, offsets, grads);
    }
    ParamType type() {
        return kCacheTable;
    }
    version_t *ver;
};

} // namespace ps
