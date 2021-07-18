#include "allocator.h"
#include "absl/types/optional.h"
#include <set>
#include <deque>
// size
// requested_size
// allocation_id
// ptr
// pre_ptr
// next_ptr
// bin_num

class BFCAllocator : public Allocator {
public:
    BFCAllocator(SubAllocator *sub_allocator, size_t total_memory,
                 bool allow_growth, const string &name,
                 bool garbage_collection = false);
    ~BFCAllocator();

    string Name() {
        return _name;
    }

    void *Allocate(size_t alignment, size_t num_bytes) override {
        return Allocate(alignment, num_bytes, AllocationAttributes());
    }

    void *Allocate(size_t alignment, size_t num_bytes,
                   const AllocationAttributes &allocation_attr) override;

    void Deallocate(void *ptr) override;

    bool TracksAllocationSizes() const override;

    size_t RequestedSize(const void *ptr) const override;

    size_t AllocatedSize(const void *ptr) const override;

    int64 AllocationId(const void *ptr) const override;

    absl::optional<AllocatorStats> GetStats() override;

    void ClearStats() override;

    void SetTimingCounter(SharedCounter *sc) {
        timing_counter_ = sc;
    }

    void SetSafeFrontier(uint64 count) override;

private:
    struct Bin;

    void *AllocateInternal(size_t alignment, size_t num_bytes,
                           bool dump_log_on_failure, uint64 freed_before_count);

    void *
    AllocateInternalWithRetry(size_t alignment, size_t num_bytes,
                              const AllocationAttributes &allocation_attr);

    void DellocateInternal(void *ptr);

    string _name;
    typedef size_t ChunkHandle;
    static const int kInvalidChunkHandle = -1;

    typedef int BinNum;
    static const int kInvalidBinNum = -1;
    // bin size : BinBaseSize*2^(i)   i = 0,1,2,3,kNumBins
    static const int kNumBins = 21;

    struct Chunk {
        // buffer size
        size_t size = 0;
        // the needed memory size for an allocation
        size_t requested_size = 0;

        // allocation_id = -1 : the chunk is unused
        int64 allocation_id = -1;

        // the pointer to the chunk memory
        void *ptr = nullptr;
        // if None, pre = -1
        // else pre is the ptr of the previous chunk
        // pre = ptr - pre->size
        ChunkHandle pre = kInvalidChunkHandle;
        // if None, next = -1
        // else next is the ptr of the next chunk
        // next = ptr + size
        ChunkHandle next = kInvalidChunkHandle;
        // The bin that this Chunk in
        BinNum bin_num = kInvalidBinNum;

        // Optional count when this Chunk was most recently made free
        uint64 freed_at_count = 0;

        bool in_use() const {
            return allocation_id != -1;
        }

        // string DebugString(){

        // }
    };

    // manage some chunks
    // chunks in this Bin sorted by chunk size
    struct Bin {
        // Chunks in this bin must chunk_size >= bin_size
        size_t bin_size = 0;

        class ChunkComparator {
        public:
            explicit ChunkComparator(BFCAllocator *allocator) :
                allocator_(allocator) {
            }
            bool operator()(const ChunkHandle ha, const ChunkHandle hb) const {
                const Chunk *a = allocator_->ChunkFromHandle(ha);
                const Chunk *b = allocator_->ChunkFromHandle(hb);
                if (a->size != b->size) {
                    return a->size < b->size;
                }
                return a->ptr < b->ptr;
            }

        private:
            BFCAllocator *allocator_;
        };

        typedef std::set<ChunkHandle, ChunkComparator> FreeChunkSet;
        // List of free chunks within the bin, sorted by chunk size
        FreeChunkSet free_chunks;
        Bin(BFCAllocator *allocator, size_t bs) :
            bin_size(bs), free_chunks(ChunkComparator(allocator)) {
        }
    };

    static const size_t kMinAllocationBits = 8;
    static const size_t kMinAllocationSize = 1 << kMinAllocationBits;

    Chunk *ChunkFromHandle(ChunkHandle h) const;

    SharedCounter *timing_counter_ = nullptr;

    static size_t RoundBytes(size_t bytes);

    // Try to add a new memory
    bool Extend(size_t aligment, size_t rounded_bytes);

    // Variable
    size_t memory_limit_ = 0;

    char bins_space_[sizeof(Bin) * kNumBins];
    //
    size_t curr_region_allocation_bytes;
    //
    size_t total_region_allocated_bytes_ = 0;
    //
    bool started_backpedal_ = false;

    bool garbage_collection_;

    std::unique_ptr<SubAllocator> sub_allocator_;

    string name_;
    SharedCounter *timing_counter_ = nullptr;
    std::deque<ChunkHandle> timestamped_chunks_;

    std::atomic<uint64> safe_frontier_ = {0};

    // Structures mutable after construction

    // RegionManager region_manager_ ;

    std::vector<Chunk> chunks_;

    // Pointer to head of linked list of free Chunks
    ChunkHandle free_chunks_list_;

    // Counter containing the next unique identifier to assign to a
    // newly-created chunk.
    int64 next_allocation_id_;

    // Stats.
    AllocatorStats stats_;
};
