#pragma once

#include <vector>
#include <algorithm>

namespace hetu {

// Implement Argsort
template <typename T>
std::vector<size_t> argsort(const T *array, size_t num) {
    std::vector<size_t> array_index(num, 0);
    for (size_t i = 0; i < num; ++i)
        array_index[i] = i;

    std::sort(
        array_index.begin(), array_index.end(),
        [array](int pos1, int pos2) { return (array[pos1] < array[pos2]); });

    return array_index;
}

/*
  Unique is used to handle key duplicate in sparse pull operation
  Unique itself is a vector that contains sorted unique data
  also has a mapping from old indices to new unique indices
*/
template <typename T>
class Unique : public std::vector<T> {
public:
    Unique(const T *data, size_t num) {
        map_indices_.resize(num);
        this->reserve(num);
        auto args = argsort(data, num);
        for (size_t i = 0; i < num; i++) {
            if (i == 0 || data[args[i]] != data[args[i - 1]]) {
                this->push_back(data[args[i]]);
            }
            map_indices_[args[i]] = this->size() - 1;
        }
    }

    inline size_t map(size_t idx) {
        return map_indices_[idx];
    }

private:
    std::vector<size_t> map_indices_;
};

} // namespace hetu
