#pragma once
#include <unordered_map>
#include <memory>
#include <utility>
#include "shared_mutex.h"

namespace ps {
/*
  thread_safe unordered_map
  use read-write lock to guaruntee concurrency
*/
template <typename _Key, typename _Tp, typename _Hash = std::hash<_Key>,
          typename _Pred = std::equal_to<_Key>,
          typename _Alloc = std::allocator<std::pair<const _Key, _Tp>>>
class threadsafe_unordered_map {
private:
    std::unordered_map<_Key, _Tp, _Hash, _Pred, _Alloc> map;
    mutable shared_mutex<4> mtx;

public:
    using map_type = std::unordered_map<_Key, _Tp, _Hash, _Pred, _Alloc>;
    using key_type = typename map_type::key_type;
    using mapped_type = typename map_type::mapped_type;
    using value_type = typename map_type::value_type;
    using hasher = typename map_type::hasher;
    using key_equal = typename map_type::key_equal;
    using allocator_type = typename map_type::allocator_type;
    using reference = typename map_type::reference;
    using const_reference = typename map_type::const_reference;
    using pointer = typename map_type::pointer;
    using const_pointer = typename map_type::const_pointer;
    using iterator = typename map_type::iterator;
    using const_iterator = typename map_type::const_iterator;
    using local_iterator = typename map_type::local_iterator;
    using const_local_iterator = typename map_type::const_local_iterator;
    using size_type = typename map_type::size_type;
    using difference_type = typename map_type::difference_type;

    threadsafe_unordered_map() = default;
    threadsafe_unordered_map(const threadsafe_unordered_map &) = delete;
    threadsafe_unordered_map(threadsafe_unordered_map &&) = default;
    threadsafe_unordered_map &
    operator=(const threadsafe_unordered_map &) = delete;
    threadsafe_unordered_map &operator=(threadsafe_unordered_map &&) = delete;
    explicit threadsafe_unordered_map(
        size_type __n, const hasher &__hf = hasher(),
        const key_equal &__eql = key_equal(),
        const allocator_type &__a = allocator_type()) :
        map(__n, __hf, __eql, __a) {
    }
    template <typename _InputIterator>
    threadsafe_unordered_map(_InputIterator __first, _InputIterator __last,
                             size_type __n = 0, const hasher &__hf = hasher(),
                             const key_equal &__eql = key_equal(),
                             const allocator_type &__a = allocator_type()) :
        map(__first, __last, __n, __hf, __eql, __a) {
    }
    threadsafe_unordered_map(const map_type &v) : map(v) {
    }
    threadsafe_unordered_map(map_type &&rv) : map(std::move(rv)) {
    }
    explicit threadsafe_unordered_map(const allocator_type &__a) : map(__a) {
    }
    threadsafe_unordered_map(const map_type &__umap,
                             const allocator_type &__a) :
        map(__umap, __a) {
    }
    threadsafe_unordered_map(map_type &&__umap, const allocator_type &__a) :
        map(std::move(__umap), __a) {
    }
    threadsafe_unordered_map(std::initializer_list<value_type> __l,
                             size_type __n = 0, const hasher &__hf = hasher(),
                             const key_equal &__eql = key_equal(),
                             const allocator_type &__a = allocator_type()) :
        map(__l, __n, __hf, __eql, __a) {
    }
    threadsafe_unordered_map(size_type __n, const allocator_type &__a) :
        threadsafe_unordered_map(__n, hasher(), key_equal(), __a) {
    }
    threadsafe_unordered_map(size_type __n, const hasher &__hf,
                             const allocator_type &__a) :
        threadsafe_unordered_map(__n, __hf, key_equal(), __a) {
    }
    template <typename _InputIterator>
    threadsafe_unordered_map(_InputIterator __first, _InputIterator __last,
                             size_type __n, const allocator_type &__a) :
        map(__first, __last, __n, __a) {
    }
    template <typename _InputIterator>
    threadsafe_unordered_map(_InputIterator __first, _InputIterator __last,
                             size_type __n, const hasher &__hf,
                             const allocator_type &__a) :
        threadsafe_unordered_map(__first, __last, __n, __hf, key_equal(), __a) {
    }
    threadsafe_unordered_map(std::initializer_list<value_type> __l,
                             size_type __n, const allocator_type &__a) :
        threadsafe_unordered_map(__l, __n, hasher(), key_equal(), __a) {
    }
    threadsafe_unordered_map(std::initializer_list<value_type> __l,
                             size_type __n, const hasher &__hf,
                             const allocator_type &__a) :
        threadsafe_unordered_map(__l, __n, __hf, key_equal(), __a) {
    }
    bool empty() const noexcept {
        s_lock<4> read_lock(mtx);
        return map.empty();
    }
    size_type size() const noexcept {
        s_lock<4> read_lock(mtx);
        return map.size();
    }
    size_type max_size() const noexcept {
        s_lock<4> read_lock(mtx);
        return map.max_size();
    }
    iterator begin() noexcept {
        x_lock<4> write_lock(mtx);
        return map.begin();
    }
    const_iterator begin() const noexcept {
        s_lock<4> read_lock(mtx);
        return map.begin();
    }
    const_iterator cbegin() const noexcept {
        s_lock<4> read_lock(mtx);
        return map.cbegin();
    }
    iterator end() noexcept {
        x_lock<4> write_lock(mtx);
        return map.end();
    }
    const_iterator end() const noexcept {
        s_lock<4> read_lock(mtx);
        return map.end();
    }
    const_iterator cend() const noexcept {
        s_lock<4> read_lock(mtx);
        return map.cend();
    }
    template <typename... _Args>
    std::pair<iterator, bool> emplace(_Args &&... __args) {
        x_lock<4> write_lock(mtx);
        return map.emplace(std::forward<_Args>(__args)...);
    }
    template <typename... _Args>
    iterator emplace_hint(const_iterator __pos, _Args &&... __args) {
        x_lock<4> write_lock(mtx);
        return map.emplace_hint(__pos, std::forward<_Args>(__args)...);
    }
    std::pair<iterator, bool> insert(const value_type &__x) {
        x_lock<4> write_lock(mtx);
        return map.insert(__x);
    }
    template <typename _Pair,
              typename = typename std::enable_if<
                  std::is_constructible<value_type, _Pair &&>::value>::type>
    std::pair<iterator, bool> insert(_Pair &&__x) {
        x_lock<4> write_lock(mtx);
        return map.insert(std::forward<_Pair>(__x));
    }
    iterator insert(const_iterator __hint, const value_type &__x) {
        x_lock<4> write_lock(mtx);
        return map.insert(__hint, __x);
    }
    template <typename _Pair,
              typename = typename std::enable_if<
                  std::is_constructible<value_type, _Pair &&>::value>::type>
    iterator insert(const_iterator __hint, _Pair &&__x) {
        x_lock<4> write_lock(mtx);
        return map.insert(__hint, std::forward<_Pair>(__x));
    }
    template <typename _InputIterator>
    void insert(_InputIterator __first, _InputIterator __last) {
        x_lock<4> write_lock(mtx);
        map.insert(__first, __last);
    }
    void insert(std::initializer_list<value_type> __l) {
        x_lock<4> write_lock(mtx);
        map.insert(__l);
    }
    iterator erase(const_iterator __position) {
        x_lock<4> write_lock(mtx);
        return map.erase(__position);
    }
    iterator erase(iterator __position) {
        x_lock<4> write_lock(mtx);
        return map.erase(__position);
    }
    size_type erase(const key_type &__x) {
        x_lock<4> write_lock(mtx);
        return map.erase(__x);
    }
    iterator erase(const_iterator __first, const_iterator __last) {
        x_lock<4> write_lock(mtx);
        return map.erase(__first, __last);
    }
    void clear() noexcept {
        x_lock<4> write_lock(mtx);
        map.clear();
    }
    void swap(map_type &__x) noexcept(noexcept(map.swap(__x._M_h))) {
        x_lock<4> write_lock(mtx);
        map.swap(__x._M_h);
    }
    hasher hash_function() const {
        s_lock<4> read_lock(mtx);
        return map.hash_function();
    }
    key_equal key_eq() const {
        s_lock<4> read_lock(mtx);
        return map.key_eq();
    }
    iterator find(const key_type &__x) {
        x_lock<4> write_lock(mtx);
        return map.find(__x);
    }
    const_iterator find(const key_type &__x) const {
        s_lock<4> read_lock(mtx);
        return map.find(__x);
    }
    size_type count(const key_type &__x) const {
        s_lock<4> read_lock(mtx);
        return map.count(__x);
    }
    std::pair<iterator, iterator> equal_range(const key_type &__x) {
        x_lock<4> write_lock(mtx);
        return map.equal_range(__x);
    }
    std::pair<const_iterator, const_iterator>
    equal_range(const key_type &__x) const {
        s_lock<4> read_lock(mtx);
        return map.equal_range(__x);
    }
    mapped_type &operator[](const key_type &__k) {
        x_lock<4> write_lock(mtx);
        return map[__k];
    }
    mapped_type &operator[](key_type &&__k) {
        x_lock<4> write_lock(mtx);
        return map[std::move(__k)];
    }
    mapped_type &at(const key_type &__k) {
        x_lock<4> write_lock(mtx);
        return map.at(__k);
    }
    const mapped_type &at(const key_type &__k) const {
        s_lock<4> read_lock(mtx);
        return map.at(__k);
    }
    size_type bucket_count() const noexcept {
        s_lock<4> read_lock(mtx);
        return map.bucket_count();
    }

    size_type max_bucket_count() const noexcept {
        s_lock<4> read_lock(mtx);
        return map.max_bucket_count();
    }
    size_type bucket_size(size_type __n) const {
        s_lock<4> read_lock(mtx);
        return map.bucket_size(__n);
    }
    size_type bucket(const key_type &__key) const {
        s_lock<4> read_lock(mtx);
        return map.bucket(__key);
    }
    local_iterator begin(size_type __n) {
        x_lock<4> write_lock(mtx);
        return map.begin(__n);
    }
    const_local_iterator begin(size_type __n) const {
        s_lock<4> read_lock(mtx);
        return map.begin(__n);
    }
    const_local_iterator cbegin(size_type __n) const {
        s_lock<4> read_lock(mtx);
        return map.cbegin(__n);
    }
    local_iterator end(size_type __n) {
        x_lock<4> write_lock(mtx);
        return map.end(__n);
    }
    const_local_iterator end(size_type __n) const {
        s_lock<4> read_lock(mtx);
        return map.end(__n);
    }
    const_local_iterator cend(size_type __n) const {
        s_lock<4> read_lock(mtx);
        return map.cend(__n);
    }
    float load_factor() const noexcept {
        s_lock<4> read_lock(mtx);
        return map.load_factor();
    }
    float max_load_factor() const noexcept {
        s_lock<4> read_lock(mtx);
        return map.max_load_factor();
    }
    void max_load_factor(float __z) {
        x_lock<4> write_lock(mtx);
        map.max_load_factor(__z);
    }
    void rehash(size_type __n) {
        x_lock<4> write_lock(mtx);
        map.rehash(__n);
    }
    void reserve(size_type __n) {
        x_lock<4> write_lock(mtx);
        map.reserve(__n);
    }
    // ----------------------------- Added function
    // ----------------------------------
    template <typename... _Args>
    const_iterator emplaceIfAbsent(const key_type &__x, _Args &&... __args) {
        x_lock<4> write_lock(mtx);
        iterator iter = map.find(__x);
        if (iter == map.end()) {
            iter = map.emplace(__x, mapped_type(std::forward<_Args>(__args)...))
                       .first;
        }
        return iter;
    }
};

} // namespace ps
