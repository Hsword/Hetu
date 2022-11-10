#include "gpu_runtime.h"

void DebugCudaMalloc(cudaError_t tmp) {
    if (tmp == cudaErrorInvalidValue) {
        std::cout << " cudaErrorInvalidValue" << std::endl;
        assert(false);
    } else if (tmp == cudaErrorMemoryAllocation) {
        std::cout << "cudaErrorMemoryAllocation" << std::endl;
        assert(false);
    } else if (tmp == cudaSuccess) {
        // std::cout<<"cudaSuccess"<<std::endl;
    } else {
        std::cout << "Unknown Type" << std::endl;
        assert(false);
    }
}

std::map<int, bool> init_free_chunk_set;

std::map<int, std::multiset<Chunk>> free_chunk_set;

std::map<int, std::map<void *, size_t>> all_malloced_chunk;

bool is_chunk_init(size_t dev_id) {
    if (init_free_chunk_set.empty()
        || init_free_chunk_set.find(dev_id) == init_free_chunk_set.end()) {
        return false;
    } else {
        return init_free_chunk_set[dev_id] == 1;
    }
}

void chunk_init(size_t dev_id) {
    bool is_chunk_empty =
        init_free_chunk_set.find(dev_id) == init_free_chunk_set.end();
    if (is_chunk_empty) {
        init_free_chunk_set.insert(std::pair<int, bool>(dev_id, false));
        free_chunk_set.insert(std::pair<int, std::multiset<Chunk>>(
            dev_id, std::multiset<Chunk>()));
        all_malloced_chunk.insert(std::pair<int, std::map<void *, size_t>>(
            dev_id, std::map<void *, size_t>()));
    }
    init_free_chunk_set[dev_id] = true;
    free_chunk_set[dev_id].clear();
    all_malloced_chunk[dev_id].clear();
}

void del_chunk(void *ptr, size_t dev_id) {
    free_chunk_set[dev_id].insert(Chunk(ptr, all_malloced_chunk[dev_id][ptr]));
}

void *find_chunk(size_t _chunk_size, size_t dev_id, bool debug) {
    std::multiset<Chunk>::iterator it;
    it = free_chunk_set[dev_id].lower_bound(Chunk(NULL, _chunk_size));
    if ((it == free_chunk_set[dev_id].end())
        || (it->chunk_size != _chunk_size)) {
        void *work_data = NULL;
        cudaSetDevice(dev_id);
        if (debug)
            DebugCudaMalloc(cudaMalloc(&work_data, _chunk_size));
        else {
            cudaError_t err = cudaMalloc(&work_data, _chunk_size);
            if (err != cudaSuccess)
                return NULL;
        }
        all_malloced_chunk[dev_id].insert(
            std::pair<void *, size_t>(work_data, _chunk_size));
        return work_data;
    } else {
        void *ans = it->ptr;
        free_chunk_set[dev_id].erase(it);
        return ans;
    }
}

void clear_chunk() {
    init_free_chunk_set.clear();
    free_chunk_set.clear();
    for (auto it = all_malloced_chunk.begin(); it != all_malloced_chunk.end(); ++it) {
        CUDA_CALL(cudaSetDevice(it->first));
        for (auto iit = it->second.begin(); iit != it->second.end(); ++iit) {
            CUDA_CALL(cudaFree(iit->first));
        }
    }
    all_malloced_chunk.clear();
}
