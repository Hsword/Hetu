#include <cassert>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
#include <type_traits>
#include <sys/time.h>
#include <omp.h>

#include "dnnl.hpp"

#include "../common/c_runtime_api.h"
#include "dnnl_runtime.h"
using namespace dnnl;
using namespace std;

int arrSize(const DLArrayHandle array) {
    int num = 1;
    for (int i = 0; i < array->ndim; ++i) {
        num *= array->shape[i];
    }
    return num;
}

extern "C" int cpu_IndexedSlices2Dense(const DLArrayHandle indices,
                                       const DLArrayHandle values,
                                       DLArrayHandle output) {
    int num = arrSize(indices);
    assert(output->ndim == 2);
    int width = output->shape[1];
    int osize = output->shape[0] * width;

    float *output_data = (float *)(output->data);
    const int *indices_data = (const int *)(indices->data);
    const float *values_data = (const float *)(values->data);

#pragma omp parallel for
    for (size_t i = 0; i < osize; ++i) {
        output_data[i] = 0;
    }

#pragma omp parallel for
    for (size_t i = 0; i < num; ++i) {
        if (indices_data[i] < 0)
            continue;
        size_t dst_offset = indices_data[i] * width;
        size_t src_offset = i * width;
        for (size_t j = 0; j < width; ++j) {
            output_data[dst_offset + j] = values_data[src_offset + j];
        }
    }
}

extern "C" int cpu_AddL2Regularization(const DLArrayHandle param,
                                       const DLArrayHandle grad, float l2reg) {
    int num = arrSize(param);

    float *param_data = (float *)(param->data);
    float *grad_data = (float *)(grad->data);

#pragma omp parallel for
    for (int i = 0; i < num; ++i)
        grad_data[i] += l2reg * param_data[i];
    return 0;
}

extern "C" int cpu_SparseAddToDense(const DLArrayHandle indices,
                                    const DLArrayHandle values,
                                    DLArrayHandle output) {
    size_t num = arrSize(indices);
    assert(output->ndim == 2);
    size_t nrow = output->shape[0];
    size_t width = output->shape[1];

    float *output_data = (float *)(output->data);
    const int *indices_data = (const int *)(indices->data);
    const float *value_data = (const float *)(values->data);

#pragma omp parallel for
    for (size_t i = 0; i < num; ++i) {
        int index = indices_data[i];
        if (index < 0 || index >= nrow)
            continue;
        float *cur_output_data = output_data + index * width;
        const float *cur_input_data = value_data + i * width;
        for (size_t j = 0; j < width; ++j) {
            cur_output_data[j] += cur_input_data[j];
        }
    }
    return 0;
}

extern "C" int cpu_SGDOptimizerUpdate(const DLArrayHandle param,
                                      const DLArrayHandle grad,
                                      float learning_rate) {
    int num = arrSize(param);

    float *param_data = (float *)(param->data);
    float *grad_data = (float *)(grad->data);

#pragma omp parallel for
    for (int i = 0; i < num; i++)
        param_data[i] -= learning_rate * grad_data[i];
    return 0;
}

extern "C" int cpu_SGDOptimizerSparseUpdate(DLArrayHandle param,
                                            const DLArrayHandle indices,
                                            const DLArrayHandle values,
                                            float lr) {
    size_t num = arrSize(indices);
    assert(param->ndim == 2);
    size_t width = param->shape[1];

    float *param_data = (float *)(param->data);
    const int *indices_data = (const int *)(indices->data);
    const float *value_data = (const float *)(values->data);

#pragma omp parallel for
    for (size_t i = 0; i < num; ++i) {
        if (indices_data[i] < 0)
            continue;
        size_t dst_offset = indices_data[i] * width;
        size_t src_offset = i * width;
        for (size_t j = 0; j < width; ++j) {
            param_data[dst_offset + j] -= lr * value_data[src_offset + j];
        }
    }
    return 0;
}

extern "C" int cpu_MomentumOptimizerUpdate(DLArrayHandle param,
                                           const DLArrayHandle grad,
                                           DLArrayHandle velocity,
                                           float learning_rate, float momentum,
                                           bool nesterov) {
    int num = arrSize(param);

    float *param_data = (float *)(param->data);
    float *grad_data = (float *)(grad->data);
    float *velocity_data = (float *)(velocity->data);

    if (nesterov) {
#pragma omp parallel for
        for (int i = 0; i < num; i++) {
            velocity_data[i] =
                momentum * (velocity_data[i] - learning_rate * grad_data[i]);
            param_data[i] =
                param_data[i] + velocity_data[i] - learning_rate * grad_data[i];
        }
    } else {
#pragma omp parallel for
        for (int i = 0; i < num; i++) {
            velocity_data[i] =
                momentum * velocity_data[i] - learning_rate * grad_data[i];
            param_data[i] = param_data[i] + velocity_data[i];
        }
    }
    return 0;
}

extern "C" int cpu_AdaGradOptimizerUpdate(DLArrayHandle param,
                                          const DLArrayHandle grad,
                                          DLArrayHandle acc,
                                          float learning_rate, float eps) {
    int num = arrSize(param);

    float *param_data = (float *)(param->data);
    float *grad_data = (float *)(grad->data);
    float *acc_data = (float *)(acc->data);

#pragma omp parallel for
    for (int i = 0; i < num; i++) {
        acc_data[i] = acc_data[i] + grad_data[i] * grad_data[i];
        param_data[i] =
            param_data[i]
            - learning_rate * grad_data[i] / (sqrtf(acc_data[i]) + eps);
    }
    return 0;
}

extern "C" int cpu_AdaGradOptimizerSparseUpdate(DLArrayHandle param,
                                                const DLArrayHandle indices,
                                                const DLArrayHandle values,
                                                DLArrayHandle acc,
                                                float learning_rate,
                                                float eps) {
    size_t num = arrSize(indices);
    assert(param->ndim == 2);
    size_t width = param->shape[1];

    float *param_data = (float *)(param->data);
    const int *indices_data = (const int *)(indices->data);
    const float *value_data = (const float *)(values->data);
    float *acc_data = (float *)(acc->data);

#pragma omp parallel for
    for (size_t i = 0; i < num; ++i) {
        if (indices_data[i] < 0)
            continue;
        size_t dst_offset = indices_data[i] * width;
        size_t src_offset = i * width;
        for (size_t j = 0; j < width; ++j) {
            float cur_grad = value_data[src_offset + j];
            float cur_acc = acc_data[dst_offset + j] + cur_grad * cur_grad;
            acc_data[dst_offset + j] = cur_acc;
            param_data[dst_offset + j] -=
                learning_rate * cur_grad / (sqrtf(cur_acc) + eps);
        }
    }
    return 0;
}

extern "C" int cpu_BetatsUpdate(DLArrayHandle betats, float beta1,
                                float beta2) {
    float *betats_data = (float *)(betats->data);
    betats_data[0] *= beta1;
    betats_data[1] *= beta2;
    return 0;
}

extern "C" int
cpu_AdamOptimizerUpdate(DLArrayHandle param, const DLArrayHandle grad,
                        DLArrayHandle expavg, DLArrayHandle expavgsq,
                        DLArrayHandle maxv, float learning_rate, float beta1,
                        float beta2, DLArrayHandle betats, float eps) {
    int num = 1;
    for (int i = 0; i < param->ndim; i++)
        num *= param->shape[i];

    float *param_data = (float *)(param->data);
    float *grad_data = (float *)(grad->data);
    float *expavg_data = (float *)(expavg->data);
    float *expavgsq_data = (float *)(expavgsq->data);
    float *betats_data = (float *)(betats->data);
    float beta1t = betats_data[0], beta2t = betats_data[1];

    if (maxv != NULL) {
        float *maxv_data = (float *)(maxv->data);
#pragma omp parallel for
        for (int i = 0; i < num; i++) {
            expavg_data[i] =
                beta1 * expavg_data[i] + (1 - beta1) * grad_data[i];
            expavgsq_data[i] = beta2 * expavgsq_data[i]
                               + (1 - beta2) * grad_data[i] * grad_data[i];
            float v_local = expavgsq_data[i] / (1 - beta2t);
            float cur_maxv = max(v_local, maxv_data[i]);
            maxv_data[i] = cur_maxv;
            param_data[i] = param_data[i]
                            - learning_rate * (expavg_data[i] / (1 - beta1t))
                                  / (sqrtf(cur_maxv) + eps);
        }
    } else {
#pragma omp parallel for
        for (int i = 0; i < num; i++) {
            expavg_data[i] =
                beta1 * expavg_data[i] + (1 - beta1) * grad_data[i];
            expavgsq_data[i] = beta2 * expavgsq_data[i]
                               + (1 - beta2) * grad_data[i] * grad_data[i];
            param_data[i] =
                param_data[i]
                - learning_rate * (expavg_data[i] / (1 - beta1t))
                      / (sqrtf(expavgsq_data[i] / (1 - beta2t)) + eps);
        }
    }
    return 0;
}

extern "C" int
cpu_AdamOptimizerSparseUpdate(DLArrayHandle param, const DLArrayHandle indices,
                              const DLArrayHandle values, DLArrayHandle expavg,
                              DLArrayHandle expavgsq, DLArrayHandle maxv,
                              float learning_rate, float beta1, float beta2,
                              DLArrayHandle betats, float eps) {
    size_t num = arrSize(indices);
    assert(param->ndim == 2);
    size_t width = param->shape[1];

    float *param_data = (float *)(param->data);
    const int *indices_data = (const int *)(indices->data);
    const float *value_data = (const float *)(values->data);
    float *expavg_data = (float *)(expavg->data);
    float *expavgsq_data = (float *)(expavgsq->data);
    float *betats_data = (float *)(betats->data);
    float beta1t = betats_data[0], beta2t = betats_data[1];

    if (maxv != NULL) {
        float *maxv_data = (float *)(maxv->data);
#pragma omp parallel for
        for (size_t i = 0; i < num; ++i) {
            if (indices_data[i] < 0)
                continue;
            size_t dst_offset = indices_data[i] * width;
            size_t src_offset = i * width;
            for (size_t j = 0; j < width; ++j) {
                float cur_grad = value_data[src_offset + j];
                float cur_expavg = expavg_data[dst_offset + j] =
                    beta1 * expavg_data[dst_offset + j]
                    + (1 - beta1) * cur_grad;
                float cur_expavgsq = expavgsq_data[dst_offset + j] =
                    beta2 * expavgsq_data[dst_offset + j]
                    + (1 - beta2) * cur_grad * cur_grad;
                float v_local = cur_expavgsq / (1 - beta2t);
                float cur_maxv = max(v_local, maxv_data[dst_offset + j]);
                maxv_data[dst_offset + j] = cur_maxv;
                param_data[dst_offset + j] -= learning_rate
                                              * (cur_expavg / (1 - beta1t))
                                              / (sqrtf(cur_maxv) + eps);
            }
        }
    } else {
#pragma omp parallel for
        for (size_t i = 0; i < num; ++i) {
            if (indices_data[i] < 0)
                continue;
            size_t dst_offset = indices_data[i] * width;
            size_t src_offset = i * width;
            for (size_t j = 0; j < width; ++j) {
                float cur_grad = value_data[src_offset + j];
                float cur_expavg = expavg_data[dst_offset + j] =
                    beta1 * expavg_data[dst_offset + j]
                    + (1 - beta1) * cur_grad;
                float cur_expavgsq = expavgsq_data[dst_offset + j] =
                    beta2 * expavgsq_data[dst_offset + j]
                    + (1 - beta2) * cur_grad * cur_grad;
                float v_local = cur_expavgsq / (1 - beta2t);
                param_data[dst_offset + j] -= learning_rate
                                              * (cur_expavg / (1 - beta1t))
                                              / (sqrtf(v_local) + eps);
            }
        }
    }

    return 0;
}
