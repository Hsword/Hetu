#include "random.h"
#include <mutex>

std::mutex random_state_mutex;
HetuRandomState hetu_random_state(0);

int SetRandomSeed(uint64_t seed) {
    std::lock_guard<std::mutex> lock(random_state_mutex);
    hetu_random_state.seed = seed;
    return 0;
}

uint64_t GetSeed() {
    return hetu_random_state.seed;
}

uint64_t GetSeedSeqNum() {
    return hetu_random_state.seqnum;
}

int StepSeqNum(uint64_t num_minimum_calls) {
    std::lock_guard<std::mutex> lock(random_state_mutex);
    hetu_random_state.seqnum += num_minimum_calls;
    return 0;
}

HetuRandomState NewRandomState(uint64_t seqnum) {
    return HetuRandomState(hetu_random_state.seed, seqnum);
}

HetuRandomState &GetRandomState(uint64_t num_minimum_calls) {
    StepSeqNum(num_minimum_calls);
    return hetu_random_state;
}
