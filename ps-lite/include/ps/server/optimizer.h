#pragma once

#include <cmath>
#include "ps/server/param.h"

namespace ps {

template <typename V>
class Param;
template <typename V>
class Param2D;
template <typename V>
class CacheTable;

enum OptType {
    SGD,
    Momentum,
    NesterovMomentum,
    AdaGrad,
    Adam,
    None,
};

template <typename V>
class Optimizer {
public:
    virtual void ApplyDense(Param<V> &param, SArray<V> &grads);
    virtual void ApplySparse(Param2D<V> &param, SArray<size_t> &offsets,
                             SArray<V> &grads);
    virtual void ApplyCache(CacheTable<V> &param, SArray<version_t> &updates,
                            SArray<size_t> &offsets, SArray<V> &grads);
    virtual void InitStates(size_t size);
};

template <typename V>
class SGDOptimizer : public Optimizer<V> {
public:
    explicit SGDOptimizer(float learning_rate) : lr(learning_rate) {
    }

    void ApplyDense(Param<V> &param, SArray<V> &grads) {
#pragma omp parallel for num_threads(4)
        for (size_t j = 0; j < param.size(); ++j) {
            param[j] -= lr * grads[j];
        }
    }

    void ApplySparse(Param2D<V> &param, SArray<size_t> &offsets,
                     SArray<V> &grads) {
        size_t width = param.width;
#pragma omp parallel for num_threads(4)
        for (size_t j = 0; j < offsets.size(); ++j) {
            size_t src_offset = j * width;
            size_t dst_offset = offsets[j] * width;
            for (size_t k = 0; k < width; ++k) {
                param[dst_offset + k] -= lr * grads[src_offset + k];
            }
        }
    }

    void ApplyCache(CacheTable<V> &param, SArray<version_t> &updates,
                    SArray<size_t> &offsets, SArray<V> &grads) {
        size_t width = param.width;
        // #pragma omp parallel for num_threads(4)
        for (size_t j = 0; j < offsets.size(); ++j) {
            param.ver[offsets[j]] += updates[j];
            size_t src_offset = j * width;
            size_t dst_offset = offsets[j] * width;
            for (size_t k = 0; k < width; ++k) {
                param[dst_offset + k] -= lr * grads[src_offset + k];
            }
        }
    }

    void InitStates(size_t size) {
    }

private:
    float lr;
};

// Optimizers below need tests! No correctness guarantees.
template <typename V>
class MomentumOptimizer : public Optimizer<V> {
public:
    explicit MomentumOptimizer(float learning_rate, float momentum) :
        lr(learning_rate), moment(momentum) {
    }

    void ApplyDense(Param<V> &param, SArray<V> &grads) {
#pragma omp parallel for num_threads(4)
        for (size_t j = 0; j < param.size(); ++j) {
            velocity[j] = moment * velocity[j] - lr * grads[j];
            param[j] = param[j] + velocity[j];
        }
    }

    void ApplySparse(Param2D<V> &param, SArray<size_t> &offsets,
                     SArray<V> &grads) {
        size_t width = param.width;
#pragma omp parallel for num_threads(4)
        for (size_t j = 0; j < offsets.size(); ++j) {
            size_t src_offset = j * width;
            size_t dst_offset = offsets[j] * width;
            for (size_t k = 0; k < width; ++k) {
                size_t cur_src = src_offset + k;
                size_t cur_dst = dst_offset + k;
                velocity[cur_dst] =
                    moment * velocity[cur_dst] - lr * grads[cur_src];
                param[cur_dst] = param[cur_dst] + velocity[cur_dst];
            }
        }
    }

    void ApplyCache(CacheTable<V> &param, SArray<version_t> &updates,
                    SArray<size_t> &offsets, SArray<V> &grads) {
        size_t width = param.width;
        // #pragma omp parallel for num_threads(4)
        for (size_t j = 0; j < offsets.size(); ++j) {
            param.ver[offsets[j]] += updates[j];
            size_t src_offset = j * width;
            size_t dst_offset = offsets[j] * width;
            for (size_t k = 0; k < width; ++k) {
                size_t cur_src = src_offset + k;
                size_t cur_dst = dst_offset + k;
                velocity[cur_dst] =
                    moment * velocity[cur_dst] - lr * grads[cur_src];
                param[cur_dst] = param[cur_dst] + velocity[cur_dst];
            }
        }
    }

    void InitStates(size_t size) {
        velocity = new V[size]();
    }

private:
    float lr;
    float moment;
    V *velocity;
};

template <typename V>
class NesterovMomentumOptimizer : public Optimizer<V> {
public:
    explicit NesterovMomentumOptimizer(float learning_rate, float momentum) :
        lr(learning_rate), moment(momentum) {
    }

    void ApplyDense(Param<V> &param, SArray<V> &grads) {
#pragma omp parallel for num_threads(4)
        for (size_t j = 0; j < param.size(); ++j) {
            V temp = -lr * grads[j];
            velocity[j] = moment * (velocity[j] + temp);
            param[j] = param[j] + velocity[j] + temp;
        }
    }

    void ApplySparse(Param2D<V> &param, SArray<size_t> &offsets,
                     SArray<V> &grads) {
        size_t width = param.width;
#pragma omp parallel for num_threads(4)
        for (size_t j = 0; j < offsets.size(); ++j) {
            size_t src_offset = j * width;
            size_t dst_offset = offsets[j] * width;
            for (size_t k = 0; k < width; ++k) {
                size_t cur_src = src_offset + k;
                size_t cur_dst = dst_offset + k;
                V temp = -lr * grads[cur_src];
                velocity[cur_dst] = moment * (velocity[cur_dst] + temp);
                param[cur_dst] = param[cur_dst] + velocity[cur_dst] + temp;
            }
        }
    }

    void ApplyCache(CacheTable<V> &param, SArray<version_t> &updates,
                    SArray<size_t> &offsets, SArray<V> &grads) {
        size_t width = param.width;
        // #pragma omp parallel for num_threads(4)
        for (size_t j = 0; j < offsets.size(); ++j) {
            param.ver[offsets[j]] += updates[j];
            size_t src_offset = j * width;
            size_t dst_offset = offsets[j] * width;
            for (size_t k = 0; k < width; ++k) {
                size_t cur_src = src_offset + k;
                size_t cur_dst = dst_offset + k;
                V temp = -lr * grads[cur_src];
                velocity[cur_dst] = moment * (velocity[cur_dst] + temp);
                param[cur_dst] = param[cur_dst] + velocity[cur_dst] + temp;
            }
        }
    }

    void InitStates(size_t size) {
        velocity = new V[size]();
    }

private:
    float lr;
    float moment;
    V *velocity;
};

template <typename V>
class AdaGradOptimizer : public Optimizer<V> {
public:
    explicit AdaGradOptimizer(float learning_rate, float initial,
                              float epsilon) :
        lr(learning_rate),
        init(initial), eps(epsilon) {
    }

    void ApplyDense(Param<V> &param, SArray<V> &grads) {
#pragma omp parallel for num_threads(4)
        for (size_t j = 0; j < param.size(); ++j) {
            accum[j] = accum[j] + grads[j] * grads[j];
            param[j] = param[j] - lr * grads[j] / (sqrt(accum[j]) + eps);
        }
    }

    void ApplySparse(Param2D<V> &param, SArray<size_t> &offsets,
                     SArray<V> &grads) {
        size_t width = param.width;
#pragma omp parallel for num_threads(4)
        for (size_t j = 0; j < offsets.size(); ++j) {
            size_t src_offset = j * width;
            size_t dst_offset = offsets[j] * width;
            for (size_t k = 0; k < width; ++k) {
                size_t cur_src = src_offset + k;
                size_t cur_dst = dst_offset + k;
                accum[cur_dst] =
                    accum[cur_dst] + grads[cur_src] * grads[cur_src];
                param[cur_dst] =
                    param[cur_dst]
                    - lr * grads[cur_src] / (sqrt(accum[cur_dst]) + eps);
            }
        }
    }

    void ApplyCache(CacheTable<V> &param, SArray<version_t> &updates,
                    SArray<size_t> &offsets, SArray<V> &grads) {
        size_t width = param.width;
        // #pragma omp parallel for num_threads(4)
        for (size_t j = 0; j < offsets.size(); ++j) {
            param.ver[offsets[j]] += updates[j];
            size_t src_offset = j * width;
            size_t dst_offset = offsets[j] * width;
            for (size_t k = 0; k < width; ++k) {
                size_t cur_src = src_offset + k;
                size_t cur_dst = dst_offset + k;
                accum[cur_dst] =
                    accum[cur_dst] + grads[cur_src] * grads[cur_src];
                param[cur_dst] =
                    param[cur_dst]
                    - lr * grads[cur_src] / (sqrt(accum[cur_dst]) + eps);
            }
        }
    }

    void InitStates(size_t size) {
        accum = new V[size];
#pragma omp parallel for num_threads(4)
        for (size_t j = 0; j < size; ++j)
            accum[j] = init;
    }

private:
    float lr;
    float init;
    float eps;
    V *accum;
};

template <typename V>
class AdamOptimizer : public Optimizer<V> {
public:
    explicit AdamOptimizer(float learning_rate, float beta1, float beta2,
                           float epsilon) :
        lr(learning_rate),
        b1(beta1), b2(beta2), eps(epsilon) {
        b1t = 1.0;
        b2t = 1.0;
    }

    void ApplyDense(Param<V> &param, SArray<V> &grads) {
        b1t = b1t * b1;
        b2t = b2t * b2;
#pragma omp parallel for num_threads(4)
        for (size_t j = 0; j < param.size(); ++j) {
            marr[j] = b1 * marr[j] + (1 - b1) * grads[j];
            varr[j] = b2 * varr[j] + (1 - b2) * grads[j] * grads[j];
            param[j] =
                param[j]
                - lr * marr[j] / (1 - b1t) / (sqrt(varr[j] / (1 - b2t)) + eps);
        }
    }

    void ApplySparse(Param2D<V> &param, SArray<size_t> &offsets,
                     SArray<V> &grads) {
        size_t width = param.width;
#pragma omp parallel for num_threads(4)
        for (size_t j = 0; j < offsets.size(); ++j) {
            size_t src_offset = j * width;
            size_t dst_offset = offsets[j] * width;
            for (size_t k = 0; k < width; ++k) {
                size_t cur_src = src_offset + k;
                size_t cur_dst = dst_offset + k;
                marr[cur_dst] = b1 * marr[cur_dst] + (1 - b1) * grads[cur_src];
                varr[cur_dst] = b2 * varr[cur_dst]
                                + (1 - b2) * grads[cur_src] * grads[cur_src];
                param[cur_dst] =
                    param[cur_dst]
                    - lr * marr[cur_dst] / (1 - b1t)
                          / (sqrt(varr[cur_dst] / (1 - b2t)) + eps);
            }
        }
    }

    void ApplyCache(CacheTable<V> &param, SArray<version_t> &updates,
                    SArray<size_t> &offsets, SArray<V> &grads) {
        size_t width = param.width;
        // #pragma omp parallel for num_threads(4)
        for (size_t j = 0; j < offsets.size(); ++j) {
            param.ver[offsets[j]] += updates[j];
            size_t src_offset = j * width;
            size_t dst_offset = offsets[j] * width;
            for (size_t k = 0; k < width; ++k) {
                size_t cur_src = src_offset + k;
                size_t cur_dst = dst_offset + k;
                marr[cur_dst] = b1 * marr[cur_dst] + (1 - b1) * grads[cur_src];
                varr[cur_dst] = b2 * varr[cur_dst]
                                + (1 - b2) * grads[cur_src] * grads[cur_src];
                param[cur_dst] =
                    param[cur_dst]
                    - lr * marr[cur_dst] / (1 - b1t)
                          / (sqrt(varr[cur_dst] / (1 - b2t)) + eps);
            }
        }
    }

    void InitStates(size_t size) {
        marr = new V[size]();
        varr = new V[size]();
    }

private:
    float lr;
    float b1;
    float b2;
    float eps;
    float b1t;
    float b2t;
    V *marr;
    V *varr;
};

} // namespace ps
