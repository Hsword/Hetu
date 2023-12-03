#ifndef HETUSYS_SRC_SEED_H
#define HETUSYS_SRC_SEED_H

#include "c_runtime_api.h"

struct HetuRandomState {
    HetuRandomState(uint64_t seed_ = 0, uint64_t seqnum_ = 0) :
        seed(seed_), seqnum(seqnum_) {
    }

    uint64_t seed;
    uint64_t seqnum;
};

HETUSYS_EXTERN_C {
    int SetRandomSeed(uint64_t seed);
    uint64_t GetSeed();
    uint64_t GetSeedSeqNum();
    int StepSeqNum(uint64_t num_minimum_calls);
}

HetuRandomState NewRandomState(uint64_t seqnum);
HetuRandomState &GetRandomState(uint64_t num_minimum_calls);

#endif
