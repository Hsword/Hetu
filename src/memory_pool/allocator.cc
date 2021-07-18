#include "src/memory_pool/allocator.h"

string number_to_string(int64 num) {
    char str[30];
    sprintf(str, "%20lld\n", num);
    return str;
}

string AllocatorStats::DebugString() const {
    string str = "Limit:       " + number_to_string(this->byte_limit);
    str += "InUse:       " + number_to_string(this->used_bytes);
    str += "MaxInUse:    " + number_to_string(this->used_peak_bytes);
    str += "Alloc_num:   " + number_to_string(this->allocation_number);
    str += "MaxAllocSize:" + number_to_string(this->largest_allocation_size);

    return str;
}

SubAllocator::SubAllocator(const std::vector<Vistor> &alloc_visitors,
                           const std::vector<Vistor> &free_visitors) :
    alloc_visitors_(alloc_visitors),
    free_visitors_(free_visitors) {
}

void SubAllocator::VisitAlloc(void *ptr, int index, size_t num_bytes) {
    for (const auto &v : alloc_visitors_) {
        v(ptr, index, num_bytes);
    }
}

void SubAllocator::VisitFree(void *ptr, int index, size_t num_bytes) {
    for (int i = free_visitors_.size() - 1; i >= 0; --i) {
        free_visitors_[i](ptr, index, num_bytes);
    }
}
