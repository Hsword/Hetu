#ifndef HETUSYS_SRC_MEMORY_POOL_ALLOCATOR_H
#define HETUSYS_SRC_MEMORY_POOL_ALLOCATOR_H

#include <iostream>
#include <cstdio>
#include <stdlib.h>
#include <string>
#include <vector>
#include <functional>
#include "src/default/types.h"

#define string std::string

// Attribures for a single allocation call
struct AllocationAttributes {
    // if an allocation is fail, whether to retry
    bool _retry_after_failure = false;

    // whether an allocation willed be logging
    bool _allocation_will_be_logged = false;

    // timing count: use to decide whether the memory to be freed
    std::function<uint64()> *_freed_by_func = nullptr;

    AllocationAttributes() = default;

    AllocationAttributes(bool retry_after_failure,
                         bool allocation_will_be_logged,
                         std::function<uint64()> *freed_by_func) :
        _retry_after_failure(retry_after_failure),
        _allocation_will_be_logged(allocation_will_be_logged),
        _freed_by_func(freed_by_func){};
};

// Runtime statistics for a allocator
struct AllocatorStats {
    int64 allocation_number = 0;       // Number of allocation
    int64 used_bytes = 0;              // Number of used bytes
    int64 used_peak_bytes = 0;         // The peak bytes in use
    int64 largest_allocation_size = 0; // The largest single allocation bytes

    int64 reserved_bytes = 0;      // Number of bytes reserved
    int64 reserved_peak_bytes = 0; // The peak bytes of bytes reserved

    int64 byte_limit = 0;            // the max bytes user can allocate now
    int64 byte_reservable_limit = 0; // the max bytes of reserved memory

    // AllocatorStats():
    // 		allocation_number(0), used_bytes(0),
    // 		used_peak_bytes(0), largest_allocation_size(0),
    // 		reserved_bytes(0), reserved_peak_bytes(0),
    // 		byte_limit(0), byte_reservable_limit(0){};

    string DebugString() const;
};

class Allocator {
public:
    static const size_t AllocatorAlign = 64;

    // Return the name of the allocator
    virtual string Name() = 0;

    // Allocate for a block of memory
    // its size = num_bytes
    // its pointer % alignment = 0 (alignment is a power of 2)
    virtual void *Allocate(size_t alignment, size_t num_bytes) = 0;

    // with specified allocation
    virtual void *Allocate(size_t alignment, size_t num_bytes,
                           const AllocationAttributes &allocation_attr) {
        return Allocate(alignment, num_bytes);
    }

    // Deallocate a block of memory pointer
    virtual void Deallocate(void *ptr) = 0;

    // Returns true if this allocator tracks the sizes of allocations.
    // RequestedSize and AllocatedSize must be overridden if
    // TracksAllocationSizes is overridden to return true.
    virtual bool TracksAllocationSizes() const {
        return false;
    }

    // Returns true if this allocator allocates an opaque handle rather than the
    // requested number of bytes.
    //
    // This method returns false for most allocators, but may be used by
    // special-case allocators that track tensor usage. If this method returns
    // true, AllocateRaw() should be invoked for all values of `num_bytes`,
    // including 0.
    //
    // NOTE: It is the caller's responsibility to track whether an allocated
    // object is a buffer or an opaque handle. In particular, when this method
    // returns `true`, users of this allocator must not run any constructors or
    // destructors for complex objects, since there is no backing store for the
    // tensor in which to place their outputs.
    virtual bool AllocatesOpaqueHandle() const {
        return false;
    }

    // Returns the user-requested size of the data allocated at
    // 'ptr'.  Note that the actual buffer allocated might be larger
    // than requested, but this function returns the size requested by
    // the user.
    //
    // REQUIRES: TracksAllocationSizes() is true.
    //
    // REQUIRES: 'ptr!=nullptr' and points to a buffer previously
    // allocated by this allocator.
    virtual size_t RequestedSize(const void *ptr) const {
        // CHECK(false) << "allocator doesn't track sizes";
        return size_t(0);
    }

    // Returns the allocated size of the buffer at 'ptr' if known,
    // otherwise returns RequestedSize(ptr). AllocatedSize(ptr) is
    // guaranteed to be >= RequestedSize(ptr).
    //
    // REQUIRES: TracksAllocationSizes() is true.
    //
    // REQUIRES: 'ptr!=nullptr' and points to a buffer previously
    // allocated by this allocator.
    virtual size_t AllocatedSize(const void *ptr) const {
        return RequestedSize(ptr);
    }

    // Returns either 0 or an identifier assigned to the buffer at 'ptr'
    // when the buffer was returned by AllocateRaw. If non-zero, the
    // identifier differs from every other ID assigned by this
    // allocator.
    //
    // REQUIRES: TracksAllocationSizes() is true.
    //
    // REQUIRES: 'ptr!=nullptr' and points to a buffer previously
    // allocated by this allocator.
    virtual int64 AllocationId(const void *ptr) const {
        return 0;
    }

    // Returns the allocated size of the buffer at 'ptr' if known,
    // otherwise returns 0. This method can be called when
    // TracksAllocationSizes() is false, but can be extremely slow.
    //
    // REQUIRES: 'ptr!=nullptr' and points to a buffer previously
    // allocated by this allocator.
    virtual size_t AllocatedSizeSlow(const void *ptr) const {
        if (TracksAllocationSizes()) {
            return AllocatedSize(ptr);
        }
        return 0;
    }

    // Fills in 'stats' with statistics collected by this allocator.
    virtual absl::optional<AllocatorStats> GetStats() {
        return absl::nullopt;
    }

    // Clears the internal stats except for the `in_use` field.
    virtual void ClearStats() {
    }

    virtual void SetSafeFrontier(uint64 count) {
    }

    virtual ~Allocator();
};

class AllocatorWrapper : public Allocator {
private:
    Allocator *const wrapped_;

public:
    explicit AllocatorWrapper(Allocator *wrapped) : wrapped_(wrapped) {
    }

    ~AllocatorWrapper() {
    }

    // Return the wrapped allocator
    Allocator *wrapped() const {
        return wrapped_;
    }

    string Name() override {
        return wrapped_->Name();
    }

    void *Allocate(size_t alignment, size_t num_bytes) override {
        return wrapped_->Allocate(alignment, num_bytes);
    }

    void *Allocate(size_t alignment, size_t num_bytes,
                   const AllocationAttributes &allocation_attr) override {
        return wrapped_->Allocate(alignment, num_bytes, allocation_attr);
    }

    void Deallocate(void *ptr) override {
        wrapped_->Deallocate(ptr);
    }
    bool TracksAllocationSizes() const override {
        return wrapped_->TracksAllocationSizes();
    }

    bool AllocatesOpaqueHandle() const override {
        return wrapped_->AllocatesOpaqueHandle();
    }

    size_t RequestedSize(const void *ptr) const override {
        return wrapped_->RequestedSize(ptr);
    }

    size_t AllocatedSize(const void *ptr) const override {
        return wrapped_->AllocatedSize(ptr);
    }

    int64 AllocationId(const void *ptr) const override {
        return wrapped_->AllocationId(ptr);
    }

    size_t AllocatedSizeSlow(const void *ptr) const override {
        return wrapped_->AllocatedSizeSlow(ptr);
    }
};

// For infrequently Alloc and Free.(< Allocator)
// To experiment with cache and pool management
class SubAllocator {
public:
    // a pointer to a memory area,
    // index value for numa_node(CPU) or GPU id(GPU)
    typedef std::function<void(void *, int index, size_t)> Vistor;

    SubAllocator(const std::vector<Vistor> &alloc_visitors,
                 const std::vector<Vistor> &free_visitors);

    virtual ~SubAllocator() {
    }
    virtual void *Alloc(size_t alignment, size_t num_bytes) = 0;
    virtual void Free(void *ptr, size_t num_bytes) = 0;

protected:
    void VisitAlloc(void *ptr, int index, size_t num_bytes);

    void VisitFree(void *ptr, int index, size_t num_bytes);

    const std::vector<Vistor> alloc_visitors_;
    const std::vector<Vistor> free_visitors_;
};

#endif