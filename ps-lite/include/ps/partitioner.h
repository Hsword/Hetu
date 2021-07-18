#pragma once

#include <vector>

namespace ps {

class Partitioner {
protected:
    const std::vector<Range> &server_range;
    size_t server_num;

public:
    Partitioner() : server_range(Postoffice::Get()->GetServerKeyRanges()) {
        server_num = server_range.size();
    }
    virtual ~Partitioner() {
    }
    virtual void partitionDense(size_t length, std::vector<Key> &keys,
                                std::vector<size_t> &parts) {
    }
    virtual void partitionSparse(size_t length, size_t width,
                                 std::vector<Key> &keys,
                                 std::vector<size_t> &parts) {
    }
    virtual int queryServer(Key key) {
        return 0;
    }
};

/* Naive partitioner, average partition into servers */
class AveragePartitioner : public Partitioner {
private:
    Key _globalId;
    size_t _serverIndex;
    size_t partition_num;

public:
    AveragePartitioner(size_t part_num = 0) : Partitioner() {
        _globalId = 0;
        _serverIndex = 0;
        if (part_num == 0 || part_num > server_num)
            part_num = server_num;
        partition_num = part_num;
    }

    void partitionDense(size_t length, std::vector<Key> &keys,
                        std::vector<size_t> &parts) {
        size_t per_part_len = length / partition_num;
        size_t rem = length % partition_num;
        for (size_t i = 0; i < partition_num; i++) {
            size_t server_idx = (i + _serverIndex) % server_num;
            keys.push_back(_globalId + server_range[server_idx].begin());
            parts.push_back(per_part_len + (i < rem));
        }
        _globalId++;
        _serverIndex = (_serverIndex + partition_num) % server_num;
    }

    void partitionSparse(size_t length, size_t width, std::vector<Key> &keys,
                         std::vector<size_t> &parts) {
        partitionDense(length, keys, parts);
    }

    int queryServer(Key key) {
        size_t server_id = 0;
        while (server_id < server_num
               && key >= server_range[server_id].begin()) {
            server_id++;
        }
        return int(server_id - 1);
    }
};

/* Use blocks to partition, intuition from BytePS */
class BlockPartitioner : public Partitioner {
private:
    Key _globalId;
    size_t _serverIndex;
    size_t _block;

public:
    BlockPartitioner(size_t block_size = 1000000) : Partitioner() {
        _globalId = 0;
        _serverIndex = 0;
        _block = block_size;
    }

    void partitionDense(size_t length, std::vector<Key> &keys,
                        std::vector<size_t> &parts) {
        partitionImpl(length, _block, keys, parts);
    }

    void partitionSparse(size_t length, size_t width, std::vector<Key> &keys,
                         std::vector<size_t> &parts) {
        size_t cur_block = std::max(_block / width, size_t(1));
        partitionImpl(length, cur_block, keys, parts);
    }

    void partitionImpl(size_t length, size_t cur_block, std::vector<Key> &keys,
                       std::vector<size_t> &parts) {
        size_t DLArray_len = length;
        while (DLArray_len != 0) {
            keys.push_back(_globalId + server_range[_serverIndex].begin());
            _serverIndex++;
            auto tmp = std::min(cur_block, DLArray_len);
            parts.push_back(tmp);
            DLArray_len -= tmp;
            if (_serverIndex == server_num) {
                _globalId++;
                _serverIndex = 0;
            }
        }
    }

    int queryServer(Key key) {
        size_t server_id = 0;
        while (server_id < server_num
               && key >= server_range[server_id].begin()) {
            server_id++;
        }
        return int(server_id - 1);
    }
};

} // namespace ps
