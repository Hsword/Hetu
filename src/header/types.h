#ifndef HETUSYS_DEFAULT_TYPES_H
#define HETUSYS_DEFAULT_TYPES_H

#include <atomic>

typedef signed char int8;
typedef short int16;
typedef int int32;
typedef long long int64;

typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint32;
typedef unsigned long long uint64;

class SharedCounter {
public:
    int64 get() {
        return cnt;
    }
    int64 next() {
        return ++cnt;
    }

private:
    std::atomic<int64> cnt{0};
};

#endif
